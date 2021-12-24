from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from loguru import logger
import ray
from scipy.spatial import distance
from src.misc.rng import RngActor

def build_samples(negative_sampler_type):
    ray.init(include_dashboard=False,ignore_reinit_error=True)
    
    logger.info(f"Creating edges for {negative_sampler_type}")
    data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"
    proteins = pd.read_csv(f"{data_dir}/processed/proteins.csv")["protein"].to_list()

    folds = 10
    for fold in tqdm(range(folds),total=folds,desc="Folds"):

        fold_dir = f"{data_dir}/processed/folds/fold_{fold}"

        # load protein embedding for this fold
        protein_embedding = pd.read_csv(f"{data_dir}/processed/embeddings/protein/protein_embeddings_{fold}.csv")
        protein_embedding = protein_embedding.set_index("protein")

        sets = ["train","val","test"]

        negative_edges = {
            "train": None,
            "val": None,
            "test": None
        }

        for set in tqdm(sets,leave=False,desc="Sets"):
            # load the edges
            graph_path = f"{fold_dir}/{set}_triples.csv"
            triples = pd.read_csv(graph_path)

            # create edges to exclude
            edges_to_exlcude = pd.DataFrame(data=[],columns=["head","tail"])
            if set == "train":
                pass
            elif set == "val":
                edges_to_exlcude = negative_edges["train"].astype(str)
            elif set == "test":
                edges_to_exlcude = pd.concat([negative_edges["train"],negative_edges["val"]],ignore_index=True).astype(str)
            
            # create edges         
            positive_edges_df, negative_edges_df = _create_edges(triples,proteins,negative_sampler_type,fold,data_dir,edges_to_exlcude)
            all_edges_df = pd.concat([positive_edges_df,negative_edges_df]).sample(frac=1.0,random_state=20)

            # create edges dir
            edges_dir = f"{data_dir}/processed/edges/{negative_sampler_type}/fold_{fold}/"
            Path(edges_dir).mkdir(parents=True, exist_ok=True)

            # create labels dir
            labels_dir = f"{data_dir}/processed/edges/{negative_sampler_type}/fold_{fold}/labels/"
            Path(labels_dir).mkdir(parents=True, exist_ok=True)

            # save labels
            positive_edges_df.to_csv(f"{labels_dir}/positive_edges_{set}.csv",index=False)
            negative_edges_df.to_csv(f"{labels_dir}/negative_edges_{set}.csv",index=False)
            all_edges_df.to_csv(f"{labels_dir}/all_edges_{set}.csv",index=False)

            # build embeddings
            all_edges_embeddings_df = _create_edge_embeddings(protein_embedding,all_edges_df)

            # save embeddings
            embeddings_dir = f"{data_dir}/processed/edges/{negative_sampler_type}/fold_{fold}/embeddings/"
            Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
            all_edges_embeddings_df.to_csv(f"{embeddings_dir}/all_edge_embeddings_{set}.csv",index=False)

            # update sets
            negative_edges[set] = negative_edges_df
            

def _create_edges(triples,proteins,negative_sampler_type,fold,data_dir,edges_to_exclude):
    positive_edges = triples[["head","tail"]]
    positive_edges_df = positive_edges.copy(deep=True)
    positive_edges_df["label"] = np.ones((len(triples)))

    negative_edges_df = None
    if negative_sampler_type == "random_sampler":
        negative_edges_df = _perform_random_negative_sampling(positive_edges_df,proteins,edges_to_exclude)
    elif negative_sampler_type == "cc_sampler":
        negative_edges_df = _perform_go_negative_sampling(positive_edges_df,proteins,"cc",fold,data_dir,edges_to_exclude)
    elif negative_sampler_type == "mf_sampler":
        negative_edges_df = _perform_go_negative_sampling(positive_edges_df,proteins,"mf",fold,data_dir,edges_to_exclude)
    elif negative_sampler_type == "bp_sampler":
        negative_edges_df = _perform_go_negative_sampling(positive_edges_df,proteins,"bp",fold,data_dir,edges_to_exclude)
    else:
        raise Exception(f"Unknown sampler: {negative_sampler_type}")

    return positive_edges_df,negative_edges_df

def _perform_random_negative_sampling(positive_edges,proteins,edges_to_exclude):
    rng = np.random.default_rng(20)
    negative_edges = []

    def _sample(head):
        protein_index = rng.integers(low=0, high=len(proteins), size=1)[0]
        negative_tail = proteins[protein_index]
        negative_edge = (head,negative_tail)
        return negative_edge


    for _index, row in positive_edges.iterrows():
        head = row["head"]

        negative_edge = _sample(head)

        # check if negative edge is in edges to exclude
        head_subset = edges_to_exclude[edges_to_exclude["head"] == negative_edge[0]]
        tail_subset = head_subset[head_subset["tail"] == negative_edge[1]]

        while len(tail_subset) > 0:
            negative_edge = _sample(head)
            
            # check again
            head_subset = edges_to_exclude[edges_to_exclude["head"] == negative_edge[0]]
            tail_subset = head_subset[head_subset["tail"] == negative_edge[1]]

        negative_edges.append(negative_edge)

    negative_edges_df = pd.DataFrame(data=negative_edges,columns=["head","tail"])
    negative_edges_df["label"] = np.zeros((len(negative_edges)))

    return negative_edges_df

def _perform_go_negative_sampling(positive_edges_df,proteins,go_type,fold,data_dir,edges_to_exclude):
    # load protein embeddings in terms of go annotation
    protein_go_embeddings = pd.read_csv(f"{data_dir}/processed/embeddings/protein_go/protein_go_embeddings_{go_type}_{fold}.csv")
    protein_go_embeddings = protein_go_embeddings.set_index("protein")

    proteins_np = np.array(proteins)
    
    # ids 
    protein_go_embeddings_id = ray.put(protein_go_embeddings)
    proteins_np_id = ray.put(proteins_np)

    # rng actor
    rng_actor = RngActor.remote(seed=20)

    negative_edges_ray = []

    # for every head in edges calculate it's similarity to every other protein in network
    for item in tqdm(positive_edges_df.iterrows(),total=len(positive_edges_df),leave=False,desc=f"Performing negative sampling for : {go_type}"):
        result = _perform_go_negative_sampling_item.remote(item,protein_go_embeddings_id,proteins_np_id,rng_actor,edges_to_exclude)    
        negative_edges_ray.append(result)

    negative_edges = ray.get(negative_edges_ray)

    negative_edges_df = pd.DataFrame(data=negative_edges,columns=["head","tail"])
    negative_edges_df["label"] = np.zeros((len(negative_edges)))

    return negative_edges_df

@ray.remote
def _perform_go_negative_sampling_item(item,protein_go_embeddings,proteins_np,rng_actor,edges_to_exclude):
    row = item[1]
    head = row["head"]

    def _sample(head):
    
        # get embedding for head
        head_embedding = protein_go_embeddings[protein_go_embeddings.index == head].iloc[0].to_numpy()

        # sample a subset
        proteins_subset_idx_ref = rng_actor.get_random_numbers.remote(low=0, high=len(proteins_np), size=300)
        proteins_subset_idx = ray.get(proteins_subset_idx_ref)
        proteins_subset = proteins_np[proteins_subset_idx]
        protein_go_embeddings_subset = protein_go_embeddings[protein_go_embeddings.index.isin(proteins_subset)].copy(deep=True)

        # candidate similarities 
        protein_go_embeddings_subset["distance"] = protein_go_embeddings_subset.apply(lambda x: _calc_distance(head_embedding,x.to_numpy()),axis=1)
        protein_go_embeddings_subset = protein_go_embeddings_subset.sort_values(by=["distance"],ascending=False)

        # get one target tail
        protein_go_embeddings_subset = protein_go_embeddings_subset.reset_index()
        target_tail = protein_go_embeddings_subset["protein"].iloc[0]

        return (head,target_tail)


    negative_edge = _sample(head)

    # check if negative edge is in edges to exclude
    head_subset = edges_to_exclude[edges_to_exclude["head"] == negative_edge[0]]
    tail_subset = head_subset[head_subset["tail"] == negative_edge[1]]

    while len(tail_subset) > 0:
        negative_edge = _sample(head)
        
        # check again
        head_subset = edges_to_exclude[edges_to_exclude["head"] == negative_edge[0]]
        tail_subset = head_subset[head_subset["tail"] == negative_edge[1]]
    
    return negative_edge

def _calc_distance(head_embedding,candidate_embedding):
    sim = distance.cosine(head_embedding,candidate_embedding)
    return sim

def _create_edge_embeddings(protein_embedding_df,all_edges_df):

    protein_embedding_df_id = ray.put(protein_embedding_df)
    all_edge_embeddings_ray = [_create_edge_embedding.remote(edge,protein_embedding_df_id) for edge in all_edges_df.iterrows()]   
    all_edge_embeddings = ray.get(all_edge_embeddings_ray)

    all_edge_embeddings_df = pd.DataFrame(data=all_edge_embeddings)
    all_edge_embeddings_df["label"] = all_edges_df["label"].to_list()

    return all_edge_embeddings_df

@ray.remote
def _create_edge_embedding(item,protein_embedding_df):
    row = item[1]
    head = row["head"]
    tail = row["tail"]

    head_embedding = protein_embedding_df[protein_embedding_df.index == head].iloc[0].to_numpy()
    tail_embedding = protein_embedding_df[protein_embedding_df.index == tail].iloc[0].to_numpy()

    edge_embedding = head_embedding * tail_embedding

    return edge_embedding

if __name__ == "__main__":
    build_samples("random_sampler")
    build_samples("cc_sampler")
    build_samples("mf_sampler")
    build_samples("bp_sampler")