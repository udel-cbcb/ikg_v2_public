import data.data_utils as du
import numpy as np
from loguru import logger
import pandas as pd
import torch
import misc.utils as utils
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,make_scorer, recall_score, precision_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
def train(
        embeddings_df,
        ikg_file,
        negative_file,
        experimental_file,
        max_depth,
        criterion,
        max_features,
        min_samples_split,
        min_samples_leaf,
        n_estimators,
        subset_data
    ):

    np.random.seed(20)

    # load data
    logger.info(f"Loading IKG from : {ikg_file}")
    ikg_pd = pd.read_csv(ikg_file)

    logger.info(f"Loading experimental relations: {experimental_file}") 
    exp_relation_pd = pd.read_csv(experimental_file)

    logger.info(f"Loading negative relations")
    negative_relations_pd = pd.read_csv(negative_file)

    if subset_data == True:
        ikg_pd = ikg_pd[:500]

    # filter to keep only phosphorylations
    logger.info(f"Triples before filtering : {len(ikg_pd)}")
    # keep only phosphorylates relations
    ikg_pd = du.filter_for_phosp(ikg_pd)
    logger.info(f"Triples after filtering: {len(ikg_pd)}")    

    # create train test splits
    pos_train_triples, pos_val_triples, _ = du.create_train_val_test_splits(ikg_pd,exp_relation_pd)

    # get negative edges
    logger.info("Getting negative train triples")
    neg_train_triples = du.get_negatives(pos_train_triples,negative_relations_pd,num_negatives=1)
    
    logger.info("Getting negative val triples")
    neg_val_tripels = du.get_negatives(pos_val_triples,negative_relations_pd,num_negatives=1)

    # create labels
    positive_train_labels = np.ones((len(pos_train_triples),))
    positive_val_labels = np.ones((len(pos_val_triples),))
    negative_train_labels = np.zeros((len(neg_train_triples),))
    negative_val_labels = np.zeros((len(neg_val_tripels),))

    # combine both positive and negative
    train_edges = pos_train_triples + neg_train_triples
    val_edges = pos_val_triples + neg_val_tripels

    # combine both positive and negative labels
    train_labels = np.concatenate([positive_train_labels, negative_train_labels],axis=0)
    val_labels = np.concatenate([positive_val_labels, negative_val_labels],axis=0)

    # shuffle them
    train_edges, train_labels = shuffle(train_edges,train_labels,random_state=20)
    val_edges, val_labels = shuffle(val_edges,val_labels,random_state=20)

    # get edge embeddings
    logger.info("Getting edge embeddings")
    train_edge_embeddings = du.get_edge_embeddings(train_edges,embeddings_df)
    val_edge_embeddings = du.get_edge_embeddings(val_edges,embeddings_df)

    # create model
    logger.info("Training")
    model = RandomForestClassifier(
                        max_depth=int(np.shape(train_edge_embeddings)[1] * max_depth),
                        criterion=criterion,
                        max_features=max_features,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        n_estimators=n_estimators,
                        random_state=20)

    #model = LogisticRegression(random_state=10)

    # define the pipeline
    cv = StratifiedKFold(n_splits=10)
    steps = [("m",model)]
    pipeline = Pipeline(steps=steps)
    
    # fit
    pipeline = pipeline.fit(train_edge_embeddings,train_labels)

    # make prediction
    logger.info("Making predictions")
    labels_predicted = pipeline.predict(val_edge_embeddings)
    labels_predicted_proba = pipeline.predict_proba(val_edge_embeddings)

    # score
    scores = {
        "f1" : f1_score(val_labels,labels_predicted,average="binary",pos_label=1),
        "recall" : recall_score(val_labels,labels_predicted,average="binary",pos_label=1),
        "precision" : precision_score(val_labels,labels_predicted,average="binary",pos_label=1),
        "au_roc": roc_auc_score(val_labels,labels_predicted_proba[:,1]),
        "au_pr": average_precision_score(val_labels,labels_predicted_proba[:,1],pos_label=1)
    }

    return scores


def test():
    
    # load embeddings file
    embeddings_df = pd.read_csv("results/kg/entity_embeddings.csv")

    scores = train(
        embeddings_df=embeddings_df,
        ikg_file="data/processed/kg/ikg.csv",
        negative_file="data/processed/kg/negative_edges.csv",
        experimental_file="data/processed/kg/experimental_triples.csv",
        max_depth=0.8,
        criterion="gini",
        max_features="auto",
        min_samples_split=60,
        min_samples_leaf=40,
        n_estimators=100,
        subset_data=False
    )

    print(scores)





    


