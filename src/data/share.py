import pandas as pd
from pathlib import Path
from src.data import kg_builder
from src.misc import graph_utils as gu
from tqdm import tqdm

data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"

# load edges
ptm_df = pd.read_csv(f"{data_dir}/processed/ptm_graph.csv")
ptm_graph = gu.triples_to_graph(ptm_df)

# load go annotations
protein_to_go_mapping_df = pd.read_csv(f"{data_dir}/processed/go/proteins_to_go_mapping.csv")

proteins = list(ptm_graph.nodes())

print(len(proteins))

for protein in tqdm(proteins,leave=True):

    # get go terms
    go_terms_str = protein_to_go_mapping_df[protein_to_go_mapping_df["protein"] == protein].iloc[0]["go_terms"]
    
    go_terms = []
    if type(go_terms_str) == str:
        go_terms = go_terms_str.split(",")

    for go_term in go_terms:
        ptm_graph.add_edge(protein,go_term, edge_type=f"annotated_with")


# to triples
triples_pd = kg_builder.graph_to_triples(ptm_graph)

triples_pd.to_csv("ptm_graph.csv",index=False)