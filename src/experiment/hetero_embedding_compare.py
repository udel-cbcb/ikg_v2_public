import pandas as pd
import numpy as np
from pathlib import Path
import polyaxon
from sklearn.preprocessing import StandardScaler
from loguru import logger
from polyaxon import tracking
from polyaxon.tracking.contrib.scikit import log_classifier
import optuna
from tqdm import tqdm
from src.misc.notify_wh import notify_wh
import src.misc.utils as utils
import copy
from sklearn.metrics import roc_auc_score, precision_score, recall_score

tracking.init()

@notify_wh(webhook_url=utils.get_webhook_url(), channel="deep_learning")
def train():

    # load the data
    folds = 10

    # sets
    sets = ["0,20","1,22","2,24","3,26","4,28","5,30","6,32","7,34","8,36","9,38"]

    # model
    model_types = ["triple_walk_cbow",
                    "triple_walk_sg",
                    "deepwalk_sg",
                    "deepwalk_cbow",
                    "distmult",
                    "node2vec_sg",
                    "node2vec_cbow",
                    "transe"]

    # define data dir
    data_parent = f"{Path.home()}/data/ikg_v2_data"
    data_dir = f"{data_parent}/hetero_embedding"

    """
    # save the git hash of data used to run the command
    git_hash = utils.get_git_hash(data_parent)
    logger.info(f"Hash of data: {git_hash}")
    with open("data_hash.txt", "w") as text_file:
        text_file.write(git_hash)
    """

    # load proteins
    proteins = pd.read_csv(f"{data_dir}/processed/proteins.csv")["protein"].to_list()

    results = []
    for fold_n in tqdm(range(folds),disable=False):
        
        data_set = sets[fold_n].split(",")
        fold_n = data_set[0]
        seed = int(data_set[1])

        for model_type in tqdm(model_types,leave=False,disable=True):

            task_types = ["edge_pred","clf_kinase_substrate","clf_ec","clf_ec_full"]

            for task_type in task_types:

                if task_type == "edge_pred":
                    scores = _perform_edge_prediction(model_type,data_dir,fold_n,seed,proteins)
                elif task_type == "clf_kinase_substrate":
                    scores = _perform_clf_kinase_substrate(model_type,data_dir,fold_n,seed,proteins)
                elif task_type == "clf_ec":
                    scores = _perform_clf_ec(model_type,data_dir,fold_n,seed,proteins)
                elif task_type == "clf_ec_full":
                    scores = _perform_clf_ec_full(model_type,data_dir,fold_n,seed,proteins)



                logger.info(f"AU-ROC : {model_type} / fold_{fold_n} : {scores}")
                
                for score_type,score_val in scores.items():
                    results.append((fold_n,model_type,task_type,score_type,score_val))
            
 
    results_df = pd.DataFrame.from_records(data=results,columns=["fold","model_type","task_type","score_type","score_val"])
    results_df.to_csv("results.csv",index=False)

    tracking.log_dataframe(results_df,"results")

def _perform_edge_prediction(model_type,data_dir,fold_n,seed,proteins):
    # load edges
    train_edges = pd.read_csv(f"{data_dir}/processed/edges/fold_{fold_n}/all_edges_train.csv").drop(columns=["source"])
    test_edges = pd.read_csv(f"{data_dir}/processed/edges/fold_{fold_n}/all_edges_test.csv").drop(columns=["source"])

    # edges to index
    train_edges_indexed = _to_indexed(train_edges,proteins)
    test_edges_indexed = _to_indexed(test_edges,proteins)

    # to numpy
    train_edges_np = train_edges_indexed[["head","tail"]].to_numpy()
    train_labels = train_edges_indexed[["label"]].to_numpy().ravel()

    test_edges_np = test_edges_indexed[["head","tail"]].to_numpy()
    test_labels = test_edges_indexed[["label"]].to_numpy().ravel()

    logger.info(f"Training for model: {model_type} / fold: {fold_n}")

    # edge embedding
    if model_type in ["triple_walk_sg","triple_walk_cbow"]:
        protein_head_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_head_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()
        protein_tail_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_tail_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()

        # get head and tail embeding for train
        train_head_embedding = protein_head_embedding[train_edges_np[:,0]]
        train_tail_embedding = protein_tail_embedding[train_edges_np[:,1]]

        # get head and tail embeding for test
        test_head_embedding = protein_head_embedding[test_edges_np[:,0]]
        test_tail_embedding = protein_tail_embedding[test_edges_np[:,1]]
        
    else:
        protein_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()
        
        # get head and tail embeding for train
        train_head_embedding = protein_embedding[train_edges_np[:,0]]
        train_tail_embedding = protein_embedding[train_edges_np[:,1]]
        
        # get head and tail embeding for test
        test_head_embedding = protein_embedding[test_edges_np[:,0]]
        test_tail_embedding = protein_embedding[test_edges_np[:,1]]

    train_edge_embedding = train_head_embedding * train_tail_embedding
    test_edge_embedding = test_head_embedding * test_tail_embedding


    # shuffle
    from sklearn.utils import shuffle
    train_edge_embedding, train_labels = shuffle(train_edge_embedding,train_labels,random_state=seed)
    test_edge_embedding, test_labels = shuffle(test_edge_embedding,test_labels,random_state=seed)

    logger.info("Training edge predictor")
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=seed).fit(train_edge_embedding,train_labels)

    logger.info("Predicting")
    y_predicted_proba = clf.predict_proba(test_edge_embedding)
    y_predicted_disc = clf.predict(test_edge_embedding)

    # calculate avg precision
    au_roc = roc_auc_score(test_labels,y_predicted_proba[:,1])
    precision = precision_score(test_labels,y_predicted_disc)
    recall = recall_score(test_labels,y_predicted_disc)

    scores = {
        "au_roc": au_roc,
        "precision": precision,
        "recall": recall
    }

    return scores


def _perform_clf_ec(model_type,data_dir,fold_n,seed,proteins):
    data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
    labelled_path = f"{data_dir}/processed/proteins/ec_labels.csv"

    logger.info(f"Reading labelled data from: {labelled_path}")

    labelled_data_df = pd.read_csv(labelled_path)

    # node indices
    node_indices = []
    evaluation_nodes = labelled_data_df["protein"].to_list()
    for eval_node in evaluation_nodes:
        idx = proteins.index(eval_node)
        node_indices.append(idx)

    # get labels
    labels = labelled_data_df["ec_label"].to_numpy().ravel()

    # edge embedding
    if model_type in ["triple_walk_sg","triple_walk_cbow"]:
        protein_head_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_head_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()
        protein_tail_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_tail_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()

        # node embedding
        protein_embedding = _get_combined_embeding(protein_head_embedding,protein_tail_embedding)

    else:
        protein_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()
        
    # get node embedding
    node_embedding = protein_embedding[node_indices]

    # split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(node_embedding,
                                                        labels,test_size=0.25,random_state=seed,shuffle=True)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=seed).fit(X_train,y_train)

    y_predicted_proba = clf.predict_proba(X_test)
    y_predicted_disc = clf.predict(X_test)

    # calculate avg precision
    au_roc = roc_auc_score(y_test,y_predicted_proba[:,1])
    precision = precision_score(y_test,y_predicted_disc)
    recall = recall_score(y_test,y_predicted_disc)

    scores = {
        "au_roc": au_roc,
        "precision": precision,
        "recall": recall
    }

    return scores

def _perform_clf_ec_full(model_type,data_dir,fold_n,seed,proteins):
    data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
    labelled_path = f"{data_dir}/processed/proteins/ec_labels_full.csv"

    logger.info(f"Reading labelled data from: {labelled_path}")

    labelled_data_df = pd.read_csv(labelled_path)

    # node indices
    node_indices = []
    evaluation_nodes = labelled_data_df["protein"].to_list()
    for eval_node in evaluation_nodes:
        idx = proteins.index(eval_node)
        node_indices.append(idx)

    # get labels
    labels = labelled_data_df["ec_label"].to_numpy().ravel()

    # edge embedding
    if model_type in ["triple_walk_sg","triple_walk_cbow"]:
        protein_head_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_head_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()
        protein_tail_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_tail_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()

        # node embedding
        protein_embedding = protein_head_embedding

    else:
        protein_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()
        
    # get node embedding
    node_embedding = protein_embedding[node_indices]

    # split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(node_embedding,
                                                        labels,test_size=0.25,random_state=seed,shuffle=True)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=seed).fit(X_train,y_train)

    y_predicted_proba = clf.predict_proba(X_test)
    
    y_predicted_proba = clf.predict_proba(X_test)
    y_predicted_disc = clf.predict(X_test)

    # calculate avg precision
    au_roc = roc_auc_score(y_test,y_predicted_proba[:,1])
    precision = precision_score(y_test,y_predicted_disc)
    recall = recall_score(y_test,y_predicted_disc)

    scores = {
        "au_roc": au_roc,
        "precision": precision,
        "recall": recall
    }

    return scores

def _perform_clf_kinase_substrate(model_type,data_dir,fold_n,seed,proteins):
    data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
    labelled_path = f"{data_dir}/processed/proteins/kinase_substrates.csv"

    logger.info(f"Reading labelled data from: {labelled_path}")

    labelled_data_df = pd.read_csv(labelled_path)

    # node indices
    node_indices = []
    evaluation_nodes = labelled_data_df["proteins"].to_list()
    for eval_node in evaluation_nodes:
        idx = proteins.index(eval_node)
        node_indices.append(idx)

    # get labels
    labels = labelled_data_df["is_kinase"].to_numpy().ravel()

    # edge embedding
    if model_type in ["triple_walk_sg","triple_walk_cbow"]:
        protein_head_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_head_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()
        protein_tail_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_tail_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()

        # node embedding
        protein_embedding = _get_combined_embeding(protein_head_embedding,protein_tail_embedding)

    else:
        protein_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/{model_type}/fold_{fold_n}/protein_embeddings_{fold_n}.csv").drop(columns=["protein"]).to_numpy()
        
    # get node embedding
    node_embedding = protein_embedding[node_indices]

    # split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(node_embedding,
                                                        labels,test_size=0.25,random_state=seed,shuffle=True)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=seed).fit(X_train,y_train)

    y_predicted_proba = clf.predict_proba(X_test)
    
    y_predicted_proba = clf.predict_proba(X_test)
    y_predicted_disc = clf.predict(X_test)

    # calculate avg precision
    au_roc = roc_auc_score(y_test,y_predicted_proba[:,1])
    precision = precision_score(y_test,y_predicted_disc)
    recall = recall_score(y_test,y_predicted_disc)

    scores = {
        "au_roc": au_roc,
        "precision": precision,
        "recall": recall
    }

    return scores

def _get_combined_embeding(head_embedding,tail_embedding):
    return np.concatenate((head_embedding,tail_embedding),axis=1)
    #return head_embedding * tail_embedding

def _to_indexed(edges,proteins):
    edges_indexed = copy.deepcopy(edges)
    edges_indexed["head"] = edges_indexed["head"].apply(lambda x : proteins.index(x))
    edges_indexed["tail"] = edges_indexed["tail"].apply(lambda x : proteins.index(x))
    return edges_indexed

if __name__ == "__main__":
    train()
