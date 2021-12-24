import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import copy
from sklearn.metrics import roc_auc_score
import src.misc.utils as utils
from src.misc.notify_wh import notify_wh

from polyaxon import tracking
tracking.init()

@notify_wh(webhook_url=utils.get_webhook_url(), channel="deep_learning")
def predict():
    seed = 32

     # save the git hash of data used to run the command
    data_parent = f"{Path.home()}/data/ikg_v2_data"
    git_hash = utils.get_git_hash(data_parent)
    logger.info(f"Hash of data: {git_hash}")
    with open("data_hash.txt", "w") as text_file:
        text_file.write(git_hash)

    data_dir = f"{Path.home()}/data/ikg_v2_data/bioinformatics_analysis"

    # load proteins
    logger.info("Loading proteins")
    proteins = pd.read_csv(f"{data_dir}/processed/proteins.csv")["protein"].to_list()

    # load edges
    logger.info("Loading edges")
    train_edges = pd.read_csv(f"{data_dir}/processed/edges/all_edges_train.csv").drop(columns=["source"])
    test_edges = pd.read_csv(f"{data_dir}/processed/edges/all_edges_test.csv").drop(columns=["source"])
    
    logger.info("Creating edge embeddings")
    # edges to index
    train_edges_indexed = _to_indexed(train_edges,proteins)
    test_edges_indexed = _to_indexed(test_edges,proteins)

    # to numpy
    train_edges_np = train_edges_indexed[["head","tail"]].to_numpy()
    train_labels = train_edges_indexed[["label"]].to_numpy().ravel()

    test_edges_np = test_edges_indexed[["head","tail"]].to_numpy()
    test_labels = test_edges_indexed[["label"]].to_numpy().ravel()

    protein_head_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/protein_head_embeddings_6.csv").drop(columns=["protein"]).to_numpy()
    protein_tail_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/protein_tail_embeddings_6.csv").drop(columns=["protein"]).to_numpy()

    # get head and tail embeding for train
    train_head_embedding = protein_head_embedding[train_edges_np[:,0]]
    train_tail_embedding = protein_tail_embedding[train_edges_np[:,1]]

    # get head and tail embeding for test
    test_head_embedding = protein_head_embedding[test_edges_np[:,0]]
    test_tail_embedding = protein_tail_embedding[test_edges_np[:,1]]

    train_edge_embedding = train_head_embedding * train_tail_embedding
    test_edge_embedding = test_head_embedding * test_tail_embedding

    # shuffle
    from sklearn.utils import shuffle
    train_edge_embedding, train_labels = shuffle(train_edge_embedding,train_labels,random_state=seed)
    test_edge_embedding, test_labels = shuffle(test_edge_embedding,test_labels,random_state=seed)

    logger.info("HP tuning edge predictor")
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    clf = RandomForestClassifier(random_state=seed,n_jobs=6)

    parameters = {'max_depth':range(3,int(train_edge_embedding.shape[1])),
              'max_features':range(2,train_edge_embedding.shape[1] - 5),
              'min_samples_split': range(2,10,1),
              'min_samples_leaf': range(2,10,1),
              "n_estimators": range(100, 500, 10),
              "criterion": ["gini", "entropy"]
    }

    rf_grid = RandomizedSearchCV(clf,parameters, n_jobs=6, n_iter=10, scoring="roc_auc", verbose=10, random_state=seed)
    rf_grid.fit(X=train_edge_embedding[:1000], y=np.ravel(train_labels[:1000]))
    
    logger.info(f"Best params are: {rf_grid.best_params_}")

    logger.info("Training edge predictor")
    clf = RandomForestClassifier(random_state=seed,n_jobs=6,**rf_grid.best_params_)
    clf = clf.fit(X=train_edge_embedding,y=train_labels)

    logger.info("Predicting")
    y_predicted_proba = clf.predict_proba(test_edge_embedding)

    # calculate avg precision
    au_roc = roc_auc_score(test_labels,y_predicted_proba[:,1])

    logger.info(f"Training score is : {au_roc}")

    # make predictions
    make_predictions(clf,protein_head_embedding,protein_tail_embedding,proteins,data_dir)


def make_predictions(clf, protein_head_embedding, protein_tail_embedding, proteins, data_dir):
    
    # load idg kinases
    target_kinases = pd.read_csv(f"{data_dir}/processed/target_kinases.csv")["kinase"].to_list()

    # make possible edges
    possible_edges_indexed = []
    possibled_edges_named = []
    logger.info("Generating edges")
    for substrate in tqdm(proteins):
        for kinase in target_kinases:
            
            # get index
            kinase_idx = proteins.index(kinase)
            substrate_idx = proteins.index(substrate)

            # create edge tuple
            if kinase_idx != substrate_idx:

                # idx edge
                edge_idx = (kinase_idx,substrate_idx)
                possible_edges_indexed.append(edge_idx)

                # named edge
                edge_named = (kinase,substrate)
                possibled_edges_named.append(edge_named)


    # create a dataframe
    possible_edges_df = pd.DataFrame(data=possible_edges_indexed,columns=["head","tail"])

    # to numpy
    possible_edges_np = possible_edges_df.to_numpy()

    # get head and tail embedding
    head_embedding = protein_head_embedding[possible_edges_np[:,0]]
    tail_embedding = protein_tail_embedding[possible_edges_np[:,1]]

    # edge embedding
    edge_embedding = head_embedding * tail_embedding

    # make the predictions
    logger.info(f"Making predictions for {len(edge_embedding)} edges")

    predicted_proba = clf.predict_proba(edge_embedding)

    logger.info("Done prediction")

    true_proba = predicted_proba[:,1]

    # create a dataframe
    possible_edge_named_df = pd.DataFrame(data=possibled_edges_named,columns=["kinase","substrate"])
    possible_edge_named_df["proba"] = true_proba

    print(possible_edge_named_df)
    possible_edge_named_df.to_csv("predicted_edges.csv",index=False)    

    logger.info("Done")


def _to_indexed(edges,proteins):
    edges_indexed = copy.deepcopy(edges)
    edges_indexed["head"] = edges_indexed["head"].apply(lambda x : proteins.index(x))
    edges_indexed["tail"] = edges_indexed["tail"].apply(lambda x : proteins.index(x))
    return edges_indexed

if __name__ == "__main__":
    predict()
