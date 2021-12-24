from tqdm import tqdm
from pathlib import Path
import pandas as pd
import ray
from src.misc.pbar import ProgressBar
import numpy as np
import torch

def build():
    import ray
    ray.init(include_dashboard=False)

    data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"
    folds = 10
    sets = ["cc","mf","bp"]

    for set in tqdm(sets,desc=f"GO set"):

        # load go embeddings
        go_embeddings = pd.read_csv(f"{data_dir}/processed/embeddings/go/go_embeddings_{set}.csv")
        go_embeddings = go_embeddings.set_index("go_term")

        # load protein to go mapping
        protein_to_go_mapping = pd.read_csv(f"{data_dir}/processed/proteins_to_go_mapping_{set}.csv")
        protein_to_go_mapping = protein_to_go_mapping.astype(str).fillna("")

        for fold in tqdm(range(folds),leave=False,desc="Fold"):

            proteins = protein_to_go_mapping["protein"].to_list()

            # cache data
            go_embeddings_id = ray.put(go_embeddings)

            protein_go_embeddings_ray = []
            for item in protein_to_go_mapping.iterrows():
                result = _build_protein_embedding_item.remote(item,go_embeddings_id)
                protein_go_embeddings_ray.append(result)

            protein_go_embeddings = ray.get(protein_go_embeddings_ray)

            protein_go_embeddings_pd = pd.DataFrame(data=protein_go_embeddings)
            protein_go_embeddings_pd.insert(0,"protein",proteins)
            
            # save
            protein_go_embeddings_pd.to_csv(f"{data_dir}/processed/embeddings/protein_go/protein_go_embeddings_{set}_{fold}.csv",index=False)


@ray.remote
def _build_protein_embedding_item(item,go_embeddings):
    row = item[1]
    try:
        go_terms_string = row["go_terms"]
        if go_terms_string != np.nan:
            go_terms = row["go_terms"].split(",")
        else:
            go_terms = []
    except :
        print(row["go_terms"])
        raise Exception("error")

    # reduce go embeddings
    go_embedding_reduced = _reduce_go_embeddings(go_terms,go_embeddings)
    
    return go_embedding_reduced



def _reduce_go_embeddings(go_terms,go_embeddings):
    go_embeddings_reduced = None
    
    if len(go_terms) > 0:
        go_embeddings_list = []
        for go_term in go_terms:
            go_embedding_pd = go_embeddings[go_embeddings.index == go_term]
            if len(go_embedding_pd) > 0:
                go_embedding_np = go_embedding_pd.iloc[0].to_numpy()
                go_embeddings_list.append(go_embedding_np)

        if len(go_embeddings_list) > 0:
            go_embeddings_stacked = np.stack(go_embeddings_list,axis=0)
            go_embeddings_reduced = np.prod(go_embeddings_stacked,axis=0)
        else:
            go_embeddings_reduced = np.full((go_embeddings.shape[1],),0.00001)
    else:
        go_embeddings_reduced = np.full((go_embeddings.shape[1],),0.00001)

    return go_embeddings_reduced

if __name__ == "__main__":
    build()
            
