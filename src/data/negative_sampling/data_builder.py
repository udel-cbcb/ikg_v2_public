from loguru import logger
import networkx as nx
from src.data import constants
import src.data.iptmnet as iptmnet
import src.data.kg_builder as kgb
import src.data.data_utils as du
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from src.data import go
from pathlib import Path
from knockknock import slack_sender
from tqdm import tqdm
from src.data import uniprot

def build_data():

    data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"
    
    # create PTM network
    proteins = _create_create_ptm_network(data_dir)

    # create splits
    _create_splits(data_dir)

    # get ec category of kinases
    ec_mapping = uniprot.get_ec_mapping(proteins,data_dir)
    ec_mapping.to_csv(f"{data_dir}/processed/ec_mapping.csv",index=False)

    # create labels from ec category
    ec_labels = uniprot.create_ec_labels(ec_mapping)
    ec_labels.to_csv(f"{data_dir}/processed/ec_labels.csv",index=False)

    # generate go nodes parent mapping
    parent_mapping_pd = go.generate_go_parent_mapping(data_dir)
    parent_mapping_pd.to_csv(f"{data_dir}/processed/go_nodes_parent_mapping.csv",index=False)

    # generate protein to go nodes mapping
    protein_go_mapping = go.generate_term_mapping(proteins,data_dir)
    protein_go_mapping.to_csv(f"{data_dir}/processed/proteins_to_go_mapping.csv",index=False)

    # generate protein to go nodes mapping cc
    protein_go_mapping_cc = go.generate_term_mapping_for_type(protein_go_mapping,parent_mapping_pd,"cc")
    protein_go_mapping_cc.to_csv(f"{data_dir}/processed/proteins_to_go_mapping_cc.csv",index=False)

    # generate protein to go nodes mapping mf
    protein_go_mapping_mf = go.generate_term_mapping_for_type(protein_go_mapping,parent_mapping_pd,"mf")
    protein_go_mapping_mf.to_csv(f"{data_dir}/processed/proteins_to_go_mapping_mf.csv",index=False)

    # generate protein to go nodes mapping cc
    protein_go_mapping_bp = go.generate_term_mapping_for_type(protein_go_mapping,parent_mapping_pd,"bp")
    protein_go_mapping_bp.to_csv(f"{data_dir}/processed/proteins_to_go_mapping_bp.csv",index=False)
    
    # create GO CC tree
    cc_graph = go.generate_go_subgraph(data_dir,"cc")
    nx.write_gml(cc_graph,f"{data_dir}/processed/go_cc_graph.gml")

    # create labelled subset of GO CC tree
    cc_labelled_subset_df = go.create_labelled_subset(cc_graph,"GO:0016020",400)
    cc_labelled_subset_df.to_csv(f"{data_dir}/processed/go_cc_labelled_subset.csv",index=False)
    
    # create GO BP tree
    bp_graph = go.generate_go_subgraph(data_dir,"bp")
    nx.write_gml(bp_graph,f"{data_dir}/processed/go_bp_graph.gml")

    # create labelled subset of GO BP tree
    bp_labelled_subset_df = go.create_labelled_subset(bp_graph,"GO:0002376",400)
    bp_labelled_subset_df.to_csv(f"{data_dir}/processed/go_bp_labelled_subset.csv",index=False) 

    # create GO MF tree
    mf_graph = go.generate_go_subgraph(data_dir,"mf")
    nx.write_gml(mf_graph,f"{data_dir}/processed/go_mf_graph.gml")
    
    # create labelled subset of GO MF tree
    mf_labelled_subset_df = go.create_labelled_subset(mf_graph,"GO:0005215",400)
    mf_labelled_subset_df.to_csv(f"{data_dir}/processed/go_mf_labelled_subset.csv",index=False)


def _create_create_ptm_network(data_dir):
    _,iptmnet_graph,proteins = iptmnet.build_graph(data_dir)

    # save graph
    out_file = f"{data_dir}/processed/ptm_graph.csv"
    kgb.serialize(iptmnet_graph,out_file)

    # save proteins
    proteins_df = pd.DataFrame(data=proteins,columns=["protein"])
    proteins_df.to_csv(f"{data_dir}/processed/proteins.csv",index=False)

    return proteins

def _create_splits(data_dir):

    # load edges
    ptm_df = pd.read_csv(f"{data_dir}/processed/ptm_graph.csv")

    # load experimental edges
    exp_edges = du.get_experimental_triples(data_dir)

    # create splits
    folds = _create_train_val_test_splits(ptm_df,exp_edges)

    # serialize
    for index, fold in enumerate(folds):
        fold_dir = f"{data_dir}/processed/folds/fold_{index}"

        Path(fold_dir).mkdir(parents=True, exist_ok=True)

        # serialize
        _serialize_triples(fold[0],f"{fold_dir}/train_triples.csv")
        _serialize_triples(fold[1],f"{fold_dir}/val_triples.csv")
        _serialize_triples(fold[2],f"{fold_dir}/test_triples.csv")


def _create_train_val_test_splits(ptm_df,experimental_relations_df):
    
    # subset ikg_df to not include experimental relations
    logger.info(f"Before removing exp relation: {len(ptm_df)}")
    ptm_without_exp = du.filter_for_dataframe(ptm_df,experimental_relations_df)

    #ptm_without_exp = ptm_df
    logger.info(f"After removing exp relation: {len(ptm_without_exp)}")

    # get all non phosph relations
    ikg_non_phosp = ptm_without_exp[ptm_without_exp["relation"] != "p"]
    ikg_only_phosp = ptm_without_exp[ptm_without_exp["relation"] == "p"]

    import numpy as np
    triples = np.array(list(ikg_only_phosp.to_records(index=False)))

    # create train test split
    triples = shuffle(triples,random_state=10)
    
    from sklearn.model_selection import KFold
    n_folds = 10
    kf = KFold(n_splits=n_folds)

    folds = []

    logger.info(f"Splitting into folds: {n_folds}")
    for train_indices, test_indices in tqdm(kf.split(triples),total=n_folds):
        
        # create train an test triples
        train_triples = triples[train_indices]
        test_triples = triples[test_indices]
          
        # create val set
        train_triples, val_triples = train_test_split(train_triples,random_state=20,shuffle=True,test_size=0.20)

        # to list
        train_triples = train_triples.tolist()
        val_triples = val_triples.tolist()
        test_triples = test_triples.tolist()        

        # non phosph to train
        non_phosph_triples = list(ikg_non_phosp.to_records(index=False))
        train_triples.extend(non_phosph_triples)    

        # add experimental triples to test set
        triples_exp = list(experimental_relations_df.to_records(index=False))
        test_triples.extend(triples_exp)
    
        folds.append((train_triples,val_triples,test_triples))

    # print stats
    logger.info(f"Train triples: {len(folds[0][0])}")
    logger.info(f"Val triples: {len(folds[0][1])}")
    logger.info(f"Test triples: {len(folds[0][2])}")

    return folds


def _serialize_triples(triples, outfile):
    triples_df = pd.DataFrame.from_records(data=triples,columns=["head","relation","tail"])
    triples_df.to_csv(outfile,index=False)


if __name__ == "__main__":
    build_data()
