from loguru import logger
import pandas as pd
from pathos.helpers.mp_helper import random_state
from pykeen import triples
import pykeen
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen.sampling import negative_sampler
from pykeen.sampling.basic_negative_sampler import BasicNegativeSampler
from pykeen.sampling.bernoulli_negative_sampler import BernoulliNegativeSampler
from pykeen.triples import TriplesFactory, triples_factory
from pykeen.stoppers import EarlyStopper
import torch
from pykeen import utils as pykutils
from tqdm import tqdm
import numpy as np
import misc.experiment_utils as eu
from misc import utils
from model.bio_negative_sampler import BioNegativeSampler
from pykeen.evaluation import SklearnEvaluator
import data.data_utils as du
from sklearn.model_selection import train_test_split
import data.constants as constants
from sklearn.utils import shuffle
from sklearn.metrics import f1_score,make_scorer, recall_score, precision_score, average_precision_score, roc_auc_score

def train(
        embedding_dim,
        algo,
        num_epochs,
        ikg_file,
        entity_mapping_file,
        relation_mapping_file,
        experimental_relations_file,
        save_embedding=False,
        perform_testing=True,
        negative_file=None,
        subsample=False,
    ):

    pykutils.set_random_seed(20)

    # load data
    logger.info(f"Loading IKG from : {ikg_file}")
    ikg_pd = pd.read_csv(ikg_file)
    if subsample == True:
        ikg_pd = ikg_pd[:10000]

    logger.info(f"Loading entity mapping from: {entity_mapping_file}")
    entity_mapping_pd = pd.read_csv(entity_mapping_file).sort_values(by=["index"],ascending=True)

    logger.info(f"Loading relation mapping from : {relation_mapping_file}")
    relation_mapping_pd = pd.read_csv(relation_mapping_file).sort_values(by=["index"],ascending=True)

    logger.info(f"Loading experimental relations") 
    exp_relation_pd = pd.read_csv(experimental_relations_file)
    
    # create entity mapping
    entity_mapping = du.create_entity_mapping(entity_mapping_pd)    

    # create relation mapping
    relation_mapping = du.create_relation_mapping(relation_mapping_pd)

    # create train test splits
    train_triples, val_triples, _ = du.create_train_val_test_splits(ikg_pd,exp_relation_pd)

    # test triples
    train_triples, test_triples = train_test_split(train_triples,test_size=0.20,random_state=20)

    # convert to indexed triples
    train_triples_tensor = torch.from_numpy(du.to_indexed(train_triples,entity_mapping,relation_mapping)) 
    val_triples_tensor = torch.from_numpy(du.to_indexed(val_triples,entity_mapping,relation_mapping))
    test_triples_tensor = torch.from_numpy(du.to_indexed(test_triples,entity_mapping,relation_mapping))

    # shuffle
    train_triples_tensor = utils.shuffle_tensor(train_triples_tensor)
    val_triples_tensor = utils.shuffle_tensor(val_triples_tensor)
    test_triples_tensor = utils.shuffle_tensor(test_triples_tensor)

    # create factories
    create_inverse_tuples=False
    if algo == "conve":
        create_inverse_tuples=True
    
    training_factory = TriplesFactory(mapped_triples=train_triples_tensor,
                        entity_to_id=entity_mapping,
                        relation_to_id=relation_mapping,
                        create_inverse_triples=create_inverse_tuples
                    )

    testing_factory = TriplesFactory(mapped_triples=test_triples_tensor,
                        entity_to_id=entity_mapping,
                        relation_to_id=relation_mapping,
                        create_inverse_triples=create_inverse_tuples
                    )

    val_factory = TriplesFactory(mapped_triples=val_triples_tensor,
                        entity_to_id=entity_mapping,
                        relation_to_id=relation_mapping,
                        create_inverse_triples=create_inverse_tuples
                    )

    test_phosph_factory = testing_factory.new_with_restriction(relations=["phosphorylates"])
    val_phosph_factory = val_factory.new_with_restriction(relations=["phosphorylates"])

    logger.info(f"Number of phosph relations in testing: {len(test_phosph_factory.mapped_triples)}")
    logger.info(f"Number of phosph relations in validation: {len(val_phosph_factory.mapped_triples)}")

    # create the model
    from pykeen.models import RESCAL
    model = RESCAL(triples_factory=training_factory,random_seed=20,embedding_dim=embedding_dim)

    # create the optimizer
    from torch.optim import Adam
    optimizer = Adam(params=model.get_grad_params(),lr=0.01)

    # create negative sample
    negative_sampler = BasicNegativeSampler(
        triples_factory=training_factory,
    )

    # define training loop
    from pykeen.training import SLCWATrainingLoop
    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=training_factory,
        optimizer=optimizer,
        negative_sampler=negative_sampler
    )
        
    # create early stopper
    early_stopping_evaluator = RankBasedEvaluator(batch_size=512,)
    def on_stop_callback(stopper,result,epoch):
        logger.info(f"Early stopping at Epoch : {epoch}")

    early_stopper = EarlyStopper(model=model,
                                evaluator=early_stopping_evaluator,
                                training_triples_factory=training_factory,
                                evaluation_triples_factory=val_phosph_factory,
                                metric="adjusted_mean_rank_index",
                                patience=10,
                                relative_delta=0.0001,
                                frequency=10,
                                stopped_callbacks=[on_stop_callback]
                            )

    # train
    _ = training_loop.train(
        triples_factory=training_factory,
        num_epochs=num_epochs,
        batch_size=128,
        stopper=early_stopper,
        use_tqdm=True,
        use_tqdm_batch=False
    )
    
    # evaluate
    final_evaluator = RankBasedEvaluator()
    results = final_evaluator.evaluate(
            model=model,
            mapped_triples=test_phosph_factory.mapped_triples,
            batch_size=128,
            additional_filter_triples=[
                train_triples_tensor,
                val_triples_tensor
            ],
    )

    # get entitiy embeddings
    entity_embeddings_np = model.entity_representations[0](indices=None).cpu().detach().numpy()
    logger.info(f"Shape of entity embedding : {entity_embeddings_np.shape}")
    entity_embeddings_pd = pd.DataFrame(data=entity_embeddings_np)
    entity_embeddings_pd.insert(0,"entity",entity_mapping_pd["entity"])

       
    # get the results
    scores = {
        "au_roc_kg": results.get_metric("adjusted_mean_rank_index")
    }

    if perform_testing == True:
        """from model import logistic_regression_model
        test_scores = logistic_regression_model.train(
            embeddings_df=entity_embeddings_pd,
            ikg_file=ikg_file,
            negative_file=negative_file,
            experimental_file=experimental_relations_file,
            max_depth=0.8,
            criterion="entropy",
            max_features="auto",
            min_samples_split=60,
            min_samples_leaf=40,
            n_estimators=200
        )
        scores.update(test_scores)"""

        # get nagtives
        negatives_df = pd.read_csv(negative_file)
        pos_test_triples = []
        for triple in test_triples:
            if triple[1] == constants.PHOSPHORYLATES:
                pos_test_triples.append(triple)
        
        neg_test_triples = du.get_negatives(pos_test_triples,negatives_df,num_negatives=1)
        
        
        # get labels
        pos_test_labels = np.ones((len(pos_test_triples),))
        negative_test_labels = np.zeros((len(neg_test_triples),))
        
        # combine
        test_triples = pos_test_triples + neg_test_triples
        test_labels = np.concatenate([pos_test_labels, negative_test_labels],axis=0)

    
        # shuffle
        test_triples, test_labels = shuffle(test_triples,test_labels,random_state=10)

        # predict
        predictions = _perform_prediction(test_triples,model,training_factory)
        print("predictions")
        scores["f_score"] = f1_score(test_labels,predictions,average="binary",pos_label=1)
            
    return entity_embeddings_pd, scores

def _perform_prediction(test_triples, model, tf):
    predicted_labels = []
    for triple in tqdm(test_triples):
        predictions = model.get_tail_prediction_df(triple[0],triple[1],triples_factory=tf)
        is_present = _is_present(triple,predictions)
        predicted_labels.append(is_present)

    predicted_labels_np = np.asarray(predicted_labels)

    return predicted_labels_np

def _is_present(triple, predictions):
    predictions = predictions[:50]
    predictions_filtered = predictions[predictions["tail_label"] == triple[2]]
    if len(predictions_filtered)>0:
        return 1
    else:
        return 0

def test():

    embeddings_pd, score = train(
        embedding_dim=128,
        num_epochs=1000,
        algo="transe",
        ikg_file="data/processed/kg/ikg.csv",
        experimental_relations_file="data/processed/kg/experimental_triples.csv",
        entity_mapping_file="data/processed/kg/ikg_entity_map.csv",
        relation_mapping_file="data/processed/kg/ikg_relation_map.csv",
        save_embedding=False,
        subsample=True,
        perform_testing=True,
        negative_file="data/processed/kg/negative_edges.csv",
    )

    print(score)
