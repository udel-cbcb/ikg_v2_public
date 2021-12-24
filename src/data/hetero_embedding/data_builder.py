from loguru import logger
import networkx as nx
import obonet
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
import copy
import src.misc.graph_utils as gu
import ray

def build_data():
    ray.init(ignore_reinit_error=True)

    data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
    
    # create PTM network
    proteins, ptm_graph, proteins_df = _create_create_ptm_network(data_dir)

    # save ptm graph
    out_file = f"{data_dir}/processed/ptm_graph.csv"
    kgb.serialize(ptm_graph,out_file)

    # save proteins list
    proteins_df.to_csv(f"{data_dir}/processed/proteins.csv",index=False)

    # create splits
    _create_splits(data_dir)

    # generate go nodes parent mapping
    parent_mapping_pd = go.generate_go_parent_mapping(data_dir)
    parent_mapping_pd.to_csv(f"{data_dir}/processed/go/go_nodes_parent_mapping.csv",index=False)

    # create go mf subgraph
    go_mf_subraph = go.generate_go_subgraph_with_edge_type(data_dir,"mf")
    nx.write_graphml_lxml(go_mf_subraph,f"{data_dir}/processed/go/go_mf_subgraph.graphml")

    # create go bp subgraph
    go_mf_subraph = go.generate_go_subgraph_with_edge_type(data_dir,"bp")
    nx.write_graphml_lxml(go_mf_subraph,f"{data_dir}/processed/go/go_bp_subgraph.graphml")

    # create go cc subgraph
    go_mf_subraph = go.generate_go_subgraph_with_edge_type(data_dir,"cc")
    nx.write_graphml_lxml(go_mf_subraph,f"{data_dir}/processed/go/go_cc_subgraph.graphml")

    
    # generate protein to go nodes mapping
    protein_go_mapping = go.generate_term_mapping(proteins,data_dir)
    protein_go_mapping.to_csv(f"{data_dir}/processed/go/proteins_to_go_mapping.csv",index=False)

    # generate protein to go nodes mapping cc
    protein_go_mapping_cc = go.generate_term_mapping_for_type(protein_go_mapping,parent_mapping_pd,"cc")
    protein_go_mapping_cc.to_csv(f"{data_dir}/processed/go/proteins_to_go_mapping_cc.csv",index=False)

    # generate protein to go nodes mapping mf
    protein_go_mapping_mf = go.generate_term_mapping_for_type(protein_go_mapping,parent_mapping_pd,"mf")
    protein_go_mapping_mf.to_csv(f"{data_dir}/processed/go/proteins_to_go_mapping_mf.csv",index=False)

    # generate protein to go nodes mapping bp
    protein_go_mapping_bp = go.generate_term_mapping_for_type(protein_go_mapping,parent_mapping_pd,"bp")
    protein_go_mapping_bp.to_csv(f"{data_dir}/processed/go/proteins_to_go_mapping_bp.csv",index=False)

    # annotate training data
    _annotate_training_data(data_dir)


def _create_create_ptm_network(data_dir):
    _,iptmnet_graph,proteins = iptmnet.build_graph(data_dir)
    
    # save proteins
    proteins_df = pd.DataFrame(data=proteins,columns=["protein"])
    
    return proteins,iptmnet_graph,proteins_df

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


def _annotate_training_data(data_dir):

    # prelaod pro graph
    logger.info("Reading pro ontology")
    pro_graph = obonet.read_obo(f"{data_dir}/raw/misc/pro_reasoned.obo")

    # read biokg data
    logger.info(f"Reading biokg graph")
    biokg_graph = _load_biokg_graph(data_dir)
        
    folds = 10
    for fold_n in range(folds):
        train_triples_df = pd.read_csv(f"{data_dir}/processed/folds/fold_{fold_n}/train_triples.csv")
        train_graph = gu.triples_to_graph(train_triples_df)
        
        proteins = list(train_graph.nodes())

        # annotate with go mf terms
        train_graph = _annotate_with_go_terms(train_graph,proteins,"mf",data_dir)

        # annotate with go bp terms
        train_graph = _annotate_with_go_terms(train_graph,proteins,"bp",data_dir)

        # annotate with pro terms
        train_graph = _annotate_with_pro_terms(train_graph,proteins,pro_graph)

        # integrate biokg
        train_graph = _integrate_biokg(train_graph,proteins,biokg_graph)
                
        # serialize
        fold_dir = f"{data_dir}/processed/annotated/folds/fold_{fold_n}"
        Path(fold_dir).mkdir(parents=True, exist_ok=True)
        kgb.serialize(train_graph,f"{fold_dir}/train_triples.csv")

        #serialize indexed
        #kgb.serialize_indexed(train_graph,fold_dir,"train")        
        

def _annotate_with_go_terms(train_graph,proteins,term_type,data_dir):
    # load go mf graph
    go_graph = nx.read_graphml(f"{data_dir}/processed/go/go_{term_type}_subgraph.graphml")

    # load protein to go mapping
    protein_to_go_mapping_df = pd.read_csv(f"{data_dir}/processed/go/proteins_to_go_mapping_{term_type}.csv")

    # copy the train graph
    train_graph_copy = copy.deepcopy(train_graph)

    # annotate with go terms
    logger.info(f"Annotating proteins with {term_type} go terms")
    for protein in tqdm(proteins,leave=True):

        # get go terms
        go_terms_str = protein_to_go_mapping_df[protein_to_go_mapping_df["protein"] == protein].iloc[0]["go_terms"]
        
        go_terms = []
        if type(go_terms_str) == str:
            go_terms = go_terms_str.split(",")

        for go_term in go_terms:
            train_graph_copy.add_edge(protein,go_term, edge_type=f"annotated_with_{term_type}")
            train_graph_copy.add_edge(go_term,protein, edge_type=f"annotated_as_{term_type}_to")

    # merge the go graph
    logger.info(f"Merging with {term_type} graph")
    go_edges = list(go_graph.edges(data=True))
    for go_edge in tqdm(go_edges,leave=True):
        head = go_edge[0]
        tail = go_edge[1]
        relation = go_edge[2]["edge_type"]
        train_graph_copy.add_edge(head,tail,edge_type=relation)

    return train_graph_copy

def _annotate_with_pro_terms(train_graph,proteins,pro_graph):
    
    # copy the train graph
    train_graph_copy = copy.deepcopy(train_graph)

    pro_nodes = list(pro_graph.nodes(data=True))

    logger.info("Annotating with pro nodes")
    for pro_node in tqdm(pro_nodes,leave=True):
        pro_node_id = pro_node[0]

        # clean the node id
        pro_node_id_cleaned = iptmnet.clean_entity(pro_node_id)

        # add is_exactly to connect protein in iPTMnet to PRO
        if pro_node_id_cleaned in proteins:
            
            # is_exactly
            train_graph_copy.add_edge(pro_node_id_cleaned,pro_node_id,edge_type="has_a")
            train_graph_copy.add_edge(pro_node_id,pro_node_id_cleaned,edge_type="is_a")

        # add is_a relationship
        if "is_a" in pro_node[1] and "PR" in pro_node_id:
            is_a_nodes = pro_node[1]["is_a"]
            for is_a_node in is_a_nodes:
                train_graph_copy.add_edge(pro_node_id,is_a_node,edge_type="is_a")
                train_graph_copy.add_edge(is_a_node,pro_node_id,edge_type="has_a")

    
    return train_graph_copy


def _integrate_biokg(train_graph,proteins,biokg_graph):
    # copy the train graph
    train_graph_copy = copy.deepcopy(train_graph)
    
    biokg_edges = list(biokg_graph.edges(data=True))

    non_protein_relations = [
        "COMPLEX_IN_PATHWAY",
        "COMPLEX_TOP_LEVEL_PATHWAY",
        "DISEASE_GENETIC_DISORDER",
        "DISEASE_PATHWAY_ASSOCIATION"
    ]

    logger.info("Merging biokg edges")
    for edge in tqdm(biokg_edges):
        head = edge[0]
        tail = edge[1]
        relation = edge[2]["edge_type"]

        # if the head relation is not directly related to protein       
        if relation in non_protein_relations:
            train_graph_copy.add_edge(head,tail,edge_type=relation)

        # for all remaining relations if not PPI
        elif relation != "PPI":
            if head in proteins:
                train_graph_copy.add_edge(head,tail,edge_type=relation)

    return train_graph_copy

def _load_biokg_graph(datadir):
    # load the triples
    biokg_triples = pd.read_csv(f"{datadir}/raw/biokg/biokg.links.tsv",sep="\t",names=["head","relation","tail"])
    
    # convert triples to graph
    biokg_graph = gu.triples_to_graph(biokg_triples)

    return biokg_graph

if __name__ == "__main__":
    build_data()
