from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from loguru import logger
import ray
from scipy.spatial import distance
from src.misc.rng import RngActor

def _build_negative_edges():
    ray.init(include_dashboard=False,ignore_reinit_error=True)
    
    folds = 10
    
    data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
    proteins = pd.read_csv(f"{data_dir}/processed/proteins.csv")["protein"].to_list()

    
    for fold in tqdm(range(folds),total=folds,desc="Folds"):

        fold_dir = f"{data_dir}/processed/folds/fold_{fold}"

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
            
            # create edges
            edges_to_exlcude = pd.DataFrame(data=[],columns=["head","tail"])
            if set == "train":
                pass
            elif set == "val":
                edges_to_exlcude = negative_edges["train"].astype(str)
            elif set == "test":
                edges_to_exlcude = pd.concat([negative_edges["train"],negative_edges["val"]],ignore_index=True).astype(str)

            positive_edges_df, negative_edges_df = _create_edges(triples,proteins,fold,data_dir,edges_to_exlcude)
            all_edges_df = pd.concat([positive_edges_df,negative_edges_df]).sample(frac=1.0,random_state=20)
            
            # create edges dir
            edges_dir = f"{data_dir}/processed/edges/fold_{fold}/"
            Path(edges_dir).mkdir(parents=True, exist_ok=True)

              # save labels
            positive_edges_df.to_csv(f"{edges_dir}/positive_edges_{set}.csv",index=False)
            negative_edges_df.to_csv(f"{edges_dir}/negative_edges_{set}.csv",index=False)
            all_edges_df.to_csv(f"{edges_dir}/all_edges_{set}.csv",index=False)

            # update sets
            negative_edges[set] = negative_edges_df

def _create_edges(triples,proteins,fold,data_dir,edges_to_exclude):
    positive_edges = triples[["head","tail"]]
    positive_edges_df = positive_edges.copy(deep=True)
    positive_edges_df["label"] = np.ones((len(triples)))

    negative_edges_df_cc = _perform_go_cc_negative_sampling(positive_edges_df,proteins,edges_to_exclude,data_dir)
    negative_edges_negatome_df = _perform_negatome_negative_sampling(positive_edges_df,proteins,edges_to_exclude)
    
    # sample
    negative_edges_df_comb = pd.concat([negative_edges_df_cc,negative_edges_negatome_df])
    negative_edges_df = negative_edges_df_comb.sample(n=len(positive_edges_df),random_state=20,ignore_index=True)

    return positive_edges_df,negative_edges_df

def _perform_negatome_negative_sampling(positive_edges_df,proteins,edges_to_exclude):
    # load negatome
    negatome_df = pd.read_csv("/root/data/ikg_v2_data/hetero_embedding/raw/negatome/combined.txt",sep="\t",header=None,names=['proteinA', 'proteinB'])

    # id
    negatome_id = ray.put(negatome_df)
    proteins_id = ray.put(proteins)
    edges_to_exclude_id = ray.put(edges_to_exclude)

    # rng actor
    rng_actor = RngActor.remote(seed=20)

    negative_edges_ray = []

    # for every head in edges calculate it's similarity to every other protein in network
    for item in tqdm(positive_edges_df.iterrows(),total=len(positive_edges_df),leave=False,desc=f"Performing negative sampling with negatome"):
        result = _perform_negatome_negative_sampling_item.remote(item,negatome_id,proteins_id,edges_to_exclude_id,rng_actor)    
        negative_edges_ray.append(result)

    negative_edges = ray.get(negative_edges_ray)

    negative_edges_df = pd.DataFrame(data=negative_edges,columns=["head","tail"])
    
    # filter out "none"
    negative_edges_df = negative_edges_df[negative_edges_df["tail"] != "none"]
    
    negative_edges_df["label"] = np.zeros((len(negative_edges_df)))
    negative_edges_df["source"] = "negatome"

    return negative_edges_df 

    pass

@ray.remote
def _perform_negatome_negative_sampling_item(item,negatome_df,proteins,edges_to_exclude,rng_actor):

    row = item[1]
    head = row["head"]

    def _sample(head):
        # get all proteinA edges having head
        proteinA_edges = negatome_df[negatome_df["proteinA"] == head]
        num_protein_a_edges = len(proteinA_edges)

        proteinB_edges = negatome_df[negatome_df["proteinB"] == head]
        num_protein_b_edges = len(proteinB_edges)

        if num_protein_a_edges > 0:
            neg_index_remote = rng_actor.get_random_numbers.remote(low=0, high=num_protein_a_edges, size=1)
            neg_index = ray.get(neg_index_remote)[0]
            neg_tail = proteinA_edges.iloc[neg_index]["proteinB"]
            if neg_tail in proteins:
                return (head,neg_tail)
            else:
                # if we have only 1 possible edge
                if num_protein_a_edges == 0:
                    return (head,"none")

                # else loop over all edges till we find something
                else:
                    for _,proteinA_edge in proteinA_edges.iterrows():
                        neg_tail_inner = proteinA_edge["proteinB"]
                        if neg_tail_inner in proteins:
                            return (head,neg_tail_inner)
                    
                    # if nothing found return none
                    return (head,"none")

        elif num_protein_b_edges > 0:
            neg_index_remote = rng_actor.get_random_numbers.remote(low=0, high=num_protein_b_edges, size=1)
            neg_index = ray.get(neg_index_remote)[0]
            neg_tail = proteinB_edges.iloc[neg_index]["proteinA"]
            if neg_tail in proteins:
                return (head,neg_tail)
            else:
                # if we have only 1 possible edge
                if num_protein_b_edges == 0:
                    return (head,"none")

                # else loop over all edges till we find something
                else:
                    for _,proteinB_edge in proteinB_edges.iterrows():
                        neg_tail_inner = proteinB_edge["proteinA"]
                        if neg_tail_inner in proteins:
                            return (head,neg_tail_inner)
                    
                    # if nothing found return none
                    return (head,"none")
        else:
            return (head,"none")

    negative_edge = _sample(head)

    # check if negative edge is in edges to exclude
    head_subset = edges_to_exclude[edges_to_exclude["head"] == negative_edge[0]]
    tail_subset = head_subset[head_subset["tail"] == negative_edge[1]]

    times_checked = 0
    while len(tail_subset) > 0:
        
        if times_checked > 60:
            break

        negative_edge = _sample(head)
        
        # check again
        head_subset = edges_to_exclude[edges_to_exclude["head"] == negative_edge[0]]
        tail_subset = head_subset[head_subset["tail"] == negative_edge[1]]

        times_checked = times_checked + 1

    if times_checked > 0:
        negative_edge = (negative_edge[0],"none")

    return negative_edge
   

def _perform_go_cc_negative_sampling(positive_edges_df,proteins,edges_to_exclude,data_dir):
    # load protein embeddings in terms of go annotation
    protein_go_embeddings = pd.read_csv(f"{data_dir}/processed/go/protein_go_embeddings_cc.csv")
    protein_go_embeddings["protein"] = protein_go_embeddings["protein"].astype(str)
    protein_go_embeddings = protein_go_embeddings.set_index("protein")

    proteins_np = np.array(proteins)
    
    # ids
    protein_go_embeddings_id = ray.put(protein_go_embeddings)
    proteins_np_id = ray.put(proteins_np)
    edges_to_exclude_id = ray.put(edges_to_exclude)

    # rng actor
    rng_actor = RngActor.remote(seed=20)

    # ray results
    negative_edges_ray = []

    # for every head in edges calculate it's similarity to every other protein in network
    for item in tqdm(positive_edges_df.iterrows(),total=len(positive_edges_df),leave=False,desc=f"Performing negative sampling for : cc"):
        result = _perform_go_negative_sampling_item.remote(item,protein_go_embeddings_id,proteins_np_id,edges_to_exclude_id,rng_actor)    
        negative_edges_ray.append(result)

    negative_edges = ray.get(negative_edges_ray)
    negative_edges_df = pd.DataFrame(data=negative_edges,columns=["head","tail"])
    negative_edges_df["label"] = np.zeros((len(negative_edges)))
    negative_edges_df["source"] = "cc"

    return negative_edges_df

@ray.remote
def _perform_go_negative_sampling_item(item,protein_go_embeddings,proteins_np,edges_to_exclude,rng_actor):
    row = item[1]
    head = str(row["head"])
        
    def _sample(head):
        # get embedding for head
        head_embedding = protein_go_embeddings[protein_go_embeddings.index == head].iloc[0].to_numpy()

        # sample a subset
        proteins_subset_idx_ref = rng_actor.get_random_numbers.remote(low=0, high=len(proteins_np), size=300)
        proteins_subset_idx = ray.get(proteins_subset_idx_ref)
        proteins_subset = proteins_np[proteins_subset_idx].astype(str)
        protein_go_embeddings_subset = protein_go_embeddings[protein_go_embeddings.index.isin(proteins_subset)].copy(deep=True)

        # candidate similarities 
        protein_go_embeddings_subset["distance"] = protein_go_embeddings_subset.apply(lambda x: _calc_distance(head_embedding,x.to_numpy()),axis=1)
        protein_go_embeddings_subset = protein_go_embeddings_subset.sort_values(by=["distance"],ascending=False)
                
        # get one target tail
        protein_go_embeddings_subset = protein_go_embeddings_subset.reset_index()
        target_tail = str(protein_go_embeddings_subset["protein"].iloc[0])

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

def _perform_random_negative_sampling(positive_edges,proteins):
    rng = np.random.default_rng(20)
    negative_edges = []
    for index, row in positive_edges.iterrows():
        protein_index = rng.integers(low=0, high=len(proteins), size=1)[0]

        head = row["head"]
        negative_tail = proteins[protein_index]
        negative_edges.append((head,negative_tail))

    negative_edges_df = pd.DataFrame(data=negative_edges,columns=["head","tail"])
    negative_edges_df["label"] = np.zeros((len(negative_edges)))
    negative_edges_df["source"] = "random"

    return negative_edges_df

def _calc_distance(head_embedding,candidate_embedding):
    sim = distance.cosine(head_embedding,candidate_embedding)
    return sim


if __name__ == "__main__":
    _build_negative_edges()