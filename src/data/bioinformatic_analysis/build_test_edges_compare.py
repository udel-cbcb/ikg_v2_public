from pathlib import Path
import pandas as pd
from tqdm import tqdm

def build():
    data_dir = f"{Path.home()}/data/ikg_v2_data/bioinformatics_analysis"

    # load the test edges
    test_edges_df = pd.read_csv(f"{data_dir}/processed/edges/all_edges_test.csv").drop(columns=["source"])

    # load linkphinder
    linkphinder_edges_df = load_linkphinder(data_dir)
    linkphinder_edges_df = find_subset(test_edges_df,linkphinder_edges_df)
    linkphinder_edges_df.to_csv(f"{data_dir}/processed/compare/linkphinder_predicted.csv",index=False)

    # load networkin
    networkin_edges_df = load_networkin(data_dir)
    networkin_edges_df = find_subset(test_edges_df,networkin_edges_df)
    networkin_edges_df.to_csv(f"{data_dir}/processed/compare/networkin_predicted.csv",index=False)

    # load phosphopredict
    phosphopredict_edges_df = load_phosphopredict(data_dir)
    phosphopredict_edges_df = find_subset(test_edges_df,phosphopredict_edges_df)
    phosphopredict_edges_df.to_csv(f"{data_dir}/processed/compare/phosphopredict_predicted.csv",index=False)
    


def load_linkphinder(data_dir):
    # load the linkphinder edges
    linkphinder_edges_df = pd.read_csv(f"{data_dir}/raw/linkphinder/predictions-linkphinder.csv",sep="\t")
    
    # select only the subset that we want
    linkphinder_edges_df = linkphinder_edges_df[[
        "ProteinKinase_ID",
        "ProteinSubstrate_ID",
        "Score"
    ]]

    # rename columns
    linkphinder_edges_df = linkphinder_edges_df.rename(columns={
        "ProteinKinase_ID": "head",
        "ProteinSubstrate_ID": "tail",
        "Score": "score"
    })

    return linkphinder_edges_df

def load_networkin(data_dir):
    # load the linkphinder edges
    newtorkin_edges_df = pd.read_csv(f"{data_dir}/raw/linkphinder_related/predictions-related/networkin_big_data_1.1.csv")
    
    # select only the subset that we want
    newtorkin_edges_df = newtorkin_edges_df[[
        "Kinase",
        "Substrate",
        "Score"
    ]]

    # rename columns
    newtorkin_edges_df = newtorkin_edges_df.rename(columns={
        "Kinase": "head",
        "Substrate": "tail",
        "Score": "score"
    })

    return newtorkin_edges_df


def load_phosphopredict(data_dir):
    # load the linkphinder edges
    phosphopredict_edges_df = pd.read_csv(f"{data_dir}/raw/linkphinder_related/predictions-related/networkin_big_data_1.1.csv")
    
    # select only the subset that we want
    phosphopredict_edges_df = phosphopredict_edges_df[[
        "Kinase",
        "Substrate",
        "Score"
    ]]

    # rename columns
    phosphopredict_edges_df = phosphopredict_edges_df.rename(columns={
        "Kinase": "head",
        "Substrate": "tail",
        "Score": "score"
    })

    return phosphopredict_edges_df


def find_subset(test_edges_df,predicted_edges_df):

    result_tuples = []

    for _, row in tqdm(test_edges_df.iterrows(),total=len(test_edges_df)):

        # filter predicted edges
        filtered_predicted_edges = predicted_edges_df[predicted_edges_df["head"] == row["head"]]
        filtered_predicted_edges = filtered_predicted_edges[filtered_predicted_edges["tail"] == row["tail"]]

        # sort by score
        filtered_predicted_edges = filtered_predicted_edges.sort_values(by=["score"],ascending=False)

        if len(filtered_predicted_edges) != 0:
            result_item = filtered_predicted_edges.iloc[0]
            label = 0
            if result_item["score"] > 0.5:
                label = 1
            else:
                label = 0

            result_tuples.append((result_item["head"],result_item["tail"],label))
        else:
            result_tuples.append((row["head"],row["tail"],0))

    result_df = pd.DataFrame(data=result_tuples,columns=["head","tail","label"])

    return result_df

if __name__ == "__main__":
    build()

