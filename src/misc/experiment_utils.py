from sklearn.metrics import average_precision_score
from ray import tune
import pandas as pd
import os

def get_config_kg():
    return {
        "embedding_dim": tune.qrandint(32,256,16),
        "algo": tune.choice(["distmult","transe","transr","conve","rescal"]),
    }

def get_config_rf():
    return {
        "max_depth": tune.quniform(0.1,1.0,0.1),
        "criterion": tune.choice(["gini","entropy"]),
        "max_features": tune.choice(["auto","sqrt", "log2"]),
        "min_samples_split": tune.qrandint(2,100,1),
        "min_samples_leaf": tune.qrandint(2,100,1),
        "n_estimators": tune.qrandint(50,500,50),
    }

def load_best_hyper_params(exp_folder):
    latest_exp =  _get_latest_exp(exp_folder)

    results_df = pd.read_csv(latest_exp)

    scorer = "amri"

    # sort 
    results_df = results_df.sort_values(by=[scorer],ascending=False)

    # get the first row
    best_result = results_df.iloc[0]

    # build the config
    best_result_dict = best_result.to_dict()

    # extract the config
    config = _extract_config(best_result_dict)

    # get score
    au_pr_delta = best_result_dict[scorer]

    return config,au_pr_delta


def _get_latest_exp(exp_folder):
    files = os.listdir(exp_folder)
    
    # sort in descending ordert
    files.sort(reverse=True)

    latest_result = files[0]

    # get the absolute path
    latest_result = os.path.join(exp_folder,latest_result)

    return latest_result

def au_pr_delta_score(y_true,y_predicted_proba):
    # get the pr baseline
    pr_baseline = len(y_true[y_true==1]) / len(y_true)
    
    # calculate avg precision
    avg_precision = average_precision_score(y_true,y_predicted_proba)

    return avg_precision - pr_baseline

def _extract_config(results_dict):
    config = {}

    for key,value in results_dict.items():
        # check if this is config key
        if "config." in key:
            key_name = key.replace("config.","")
            config[key_name] = value

    return config

def split_proteins(proteins):
    from sklearn.model_selection import train_test_split
    proteins_train, proteins_test = train_test_split(proteins,test_size=0.3,random_state=10)
    return proteins_train,proteins_test


