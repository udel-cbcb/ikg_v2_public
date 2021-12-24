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

tracking.init()

@notify_wh(webhook_url=utils.get_webhook_url(), channel="deep_learning")
def train():
    # load the data
    folds = 10

    # sampler_type
    sampler_types = ["random_sampler","cc_sampler","bp_sampler","mf_sampler"]

    # model
    model_types = ["linear","tree"]

    results = []
    for fold_n in tqdm(range(folds),disable=True):
        for model_type in tqdm(model_types,leave=False):
            for sampler_type in tqdm(sampler_types,disable=True,leave=False):
                
                logger.info(f"Training for model: {model_type} / sampler: {sampler_type} / fold: {fold_n}")

                # load the edge embeddings
                data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"
                train_embeddings = pd.read_csv(f"{data_dir}/processed/edges/{sampler_type}/fold_{fold_n}/embeddings/all_edge_embeddings_train.csv")
                test_embeddings = pd.read_csv(f"{data_dir}/processed/edges/{sampler_type}/fold_{fold_n}/embeddings/all_edge_embeddings_test.csv")

                train_X = train_embeddings.drop(columns=["label"]).to_numpy()
                train_y = train_embeddings["label"].to_numpy()

                test_X = test_embeddings.drop(columns=["label"]).to_numpy()
                test_y = test_embeddings["label"].to_numpy().ravel()

                # scaler
                scaler = StandardScaler()
                train_X = scaler.fit_transform(train_X)
                test_X = scaler.transform(test_X)

                clf = None

                if model_type == "linear":
                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression(random_state=20)
                elif model_type == "tree":
                    from sklearn.ensemble import RandomForestClassifier
                    clf = RandomForestClassifier(random_state=20)
                else:
                    raise Exception(f"Unknown model : {model_type}")

                clf.fit(train_X,train_y)
                predicted_y = clf.predict_proba(test_X)

                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(test_y,predicted_y[:,1])

                logger.info(f"Score: {score}")

                results.append((fold_n,sampler_type,model_type,score))
 
    results_df = pd.DataFrame.from_records(data=results,columns=["fold","sampler_type","model_type","score"])
    results_df.to_csv("results.csv",index=False)

    tracking.log_dataframe(results_df,"results")

if __name__ == "__main__":
    train()
