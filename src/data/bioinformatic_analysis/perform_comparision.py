from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score


def perform():
    data_dir = f"{Path.home()}/data/ikg_v2_data/bioinformatics_analysis"

    results = []

    # load 
    test_edges= pd.read_csv(f"{data_dir}/processed/edges/all_edges_test.csv").drop(columns=["source"])["label"].to_numpy()

    # triplewalk
    triple_walk = pd.read_csv(f"{data_dir}/processed/compare/triplewalk_predicted.csv")[["label","score"]].to_numpy()
    results.extend(calc_scores("triplewalk",test_edges,triple_walk))

    # linkphinder
    linkphinder_df = pd.read_csv(f"{data_dir}/processed/compare/linkphinder_predicted.csv")[["label","score"]].to_numpy()
    results.extend(calc_scores("linkphinder",test_edges,linkphinder_df))

    # networkin
    networkin_df = pd.read_csv(f"{data_dir}/processed/compare/networkin_predicted.csv")[["label","score"]].to_numpy()
    results.extend(calc_scores("networkin",test_edges,networkin_df))

    # phosphopredict
    phosphopredict_df = pd.read_csv(f"{data_dir}/processed/compare/phosphopredict_predicted.csv")[["label","score"]].to_numpy()
    results.extend(calc_scores("phosphopredict",test_edges,phosphopredict_df))

    # netphos
    netphos_predicted_df = pd.read_csv(f"{data_dir}/processed/compare/netphos_predicted.csv")[["label","score"]].to_numpy()
    results.extend(calc_scores("netphos",test_edges,netphos_predicted_df))

    # scansite
    scansite_predicted_df = pd.read_csv(f"{data_dir}/processed/compare/scansite_predicted.csv")[["label","score"]].to_numpy()
    results.extend(calc_scores("scansite",test_edges,scansite_predicted_df))

    # to df
    results_df = pd.DataFrame(data=results,columns=["model","score_type",'score_value'])
    results_df.to_csv(f'{data_dir}/results/model_compare.csv',index=False)


def calc_scores(model,test_labels,predicted_labels):
    results = []
    
    au_roc = roc_auc_score(test_labels,predicted_labels[:,1])
    results.append((model,"au-roc",au_roc))

    precision = precision_score(test_labels,predicted_labels[:,0])
    results.append((model,"precision",precision))

    recall = recall_score(test_labels,predicted_labels[:,0])
    results.append((model,"recall",recall))

    return results




if __name__ == "__main__":
    perform()