from loguru import logger
from pathlib import Path
from numpy.random.mtrand import sample
import pandas as pd
from sklearn.preprocessing import StandardScaler
from polyaxon import tracking
import optuna
from optuna.samplers import TPESampler
import os
from src.misc.notify_wh import notify_wh
import src.misc.utils as utils
import argparse

class Objective(object):
    def __init__(self, model_type, sampler_type):
        self.model_type = model_type
        self.sampler_type = sampler_type

    def __call__(self, trial):
        fold_n = 1

        logger.info(f"Training for model: {self.model_type} / sampler: {self.sampler_type} / fold: {fold_n}")

        # load the edge embeddings
        data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"
        train_embeddings = pd.read_csv(f"{data_dir}/processed/edges/{self.sampler_type}/fold_{fold_n}/embeddings/all_edge_embeddings_train.csv")
        val_embeddings = pd.read_csv(f"{data_dir}/processed/edges/{self.sampler_type}/fold_{fold_n}/embeddings/all_edge_embeddings_val.csv")
        test_embeddings = pd.read_csv(f"{data_dir}/processed/edges/{self.sampler_type}/fold_{fold_n}/embeddings/all_edge_embeddings_test.csv")

        train_X = train_embeddings.drop(columns=["label"]).to_numpy()
        train_y = train_embeddings["label"].to_numpy()

        val_X = val_embeddings.drop(columns=["label"]).to_numpy()
        val_y = val_embeddings["label"].to_numpy().ravel()

        test_X = test_embeddings.drop(columns=["label"]).to_numpy()
        test_y = test_embeddings["label"].to_numpy().ravel()

        # scaler
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)

        clf = None

        if self.model_type == "linear":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(random_state=20,C=trial.suggest_float("C",0.7,1.0))
        elif self.model_type == "tree":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(random_state=20,
                                        n_jobs=16,
                                        n_estimators=trial.suggest_int("n_estimators", 100, 200),
                                        max_features=trial.suggest_categorical("max_features",["auto","sqrt","log2"]),
                                        min_samples_leaf=trial.suggest_int("min_samples_leaf",2,10),
                                        min_samples_split=trial.suggest_int("min_samples_split",2,10)
                                    )
        elif self.model_type == "svm":
            from sklearn.svm import SVC
            clf = SVC(random_state=20,C=trial.suggest_float("C",0.7,1.0),probability=True)
        else:
            raise Exception(f"Unknown model : {self.model_type}")

        clf.fit(train_X,train_y)
        predicted_y = clf.predict_proba(val_X)

        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(val_y,predicted_y[:,1])

        logger.info(f"Score: {score}")

        return score

@notify_wh(webhook_url=utils.get_webhook_url(), channel="deep_learning")
def start():
    tracking.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',type=str,required=True)
    parser.add_argument('--sampler_type',type=str,required=True)
    parser.add_argument("--test",type=bool)
    
    args = parser.parse_args()

    model_type = args.model_type
    sampler_type = args.sampler_type
    test=args.test

    if test == True:
        n_trials = 1
    else:
        n_trials = 100

    # create objective
    objective = Objective(model_type=model_type,sampler_type=sampler_type)

    sampler = TPESampler(seed=20)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Best params: {best_params}")
    
    best_score = study.best_value
    logger.info(f"Best score: {best_score}")

    tracking.log_outputs(best_params=best_params,best_score=best_score)

if __name__ == "__main__":
    start()
    