from src.data import uniprot
from pathlib import Path
from src.data import iptmnet
import src.data.kg_builder as kgb
import pandas as pd

def build():

    data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"

    # create PTM network
    proteins = _create_create_ptm_network(data_dir)

    # get ec category of kinases
    ec_mapping = uniprot.get_ec_mapping(proteins,data_dir)
    ec_mapping.to_csv(f"{data_dir}/processed/proteins/ec_mapping.csv",index=False)

    # create labels from ec category
    ec_labels = uniprot.create_ec_labels(ec_mapping)
    ec_labels.to_csv(f"{data_dir}/processed/proteins/ec_labels.csv",index=False)

    # get full ec category of kinases
    ec_mapping_full = uniprot.get_ec_mapping_full(proteins,data_dir)
    ec_mapping_full.to_csv(f"{data_dir}/processed/proteins/ec_mapping_full.csv",index=False)

    # create labels from full ec category
    ec_labels_full = uniprot.create_ec_labels_full(ec_mapping_full)
    ec_labels_full.to_csv(f"{data_dir}/processed/proteins/ec_labels_full.csv",index=False)

def _create_create_ptm_network(data_dir):
    _,iptmnet_graph,proteins = iptmnet.build_graph(data_dir)

    # save graph
    out_file = f"{data_dir}/processed/ptm_graph.csv"
    kgb.serialize(iptmnet_graph,out_file)

    # save proteins
    proteins_df = pd.DataFrame(data=proteins,columns=["protein"])
    proteins_df.to_csv(f"{data_dir}/processed/proteins.csv",index=False)

    return proteins

if __name__ == "__main__":
    build()