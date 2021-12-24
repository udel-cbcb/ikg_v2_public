from os import path
import re
import copy
from networkx.algorithms.shortest_paths.unweighted import predecessor
from networkx.classes.function import nodes, subgraph
from tqdm import tqdm
from loguru import logger
import obonet
import pandas as pd
import networkx as nx
import ray
from src.misc.pbar import ProgressBar

from enum import Enum
class GO_REGION(Enum):
    INTRA_CELLULAR = 1
    EXTRA_CELLULAR = 2
    NUCLEUS = 3
    CYTOPLASM = 4
    ON_CELL=5

def annotate_with_go_terms(graph,proteins):
    logger.info("Annotating with go terms")

    # make a copy
    graph_copy = copy.deepcopy(graph)

    # loop over all the nodes
    for protein in tqdm(proteins):
        # get go terms
        terms = _get_go_terms(protein)
        
        # annotate with terms
        for term in terms:
            graph_copy.add_edge(protein,term,type="annotated_with")

            # add type edge
            graph_copy.add_edge(term,"go_term",type="type_of")

    return graph_copy

def merge_with_go_graph(graph):

    # clone
    graph_copy = copy.deepcopy(graph)

    # load the go graph
    logger.info("Reading go graph")
    go_graph = obonet.read_obo("data/raw/misc/goslim_pir.obo")

    # loop over nodes
    logger.info("Merging go nodes")
    go_nodes = list(go_graph.nodes(data=True))
    for node in tqdm(go_nodes):
        node_id = node[0]

        # add is_a relations
        if "is_a" in node[1]:
            is_a_nodes = node[1]["is_a"]
            for is_a_node in is_a_nodes:
                graph_copy.add_edge(node_id,is_a_node,type="is_a")
                graph_copy.add_edge(is_a_node,"go_term",type="type_of")   
        
    return graph_copy

def generate_term_mapping(proteins,data_dir):

    mappings = []
    for protein in tqdm(proteins):
        go_terms = ",".join(_get_go_terms(protein,data_dir))
        mappings.append((protein,go_terms))

    mappings_pd = pd.DataFrame(data=mappings,columns=["protein","go_terms"])
    return mappings_pd

def generate_term_mapping_for_type(protein_to_go_mapping,go_parent_mapping,go_type):
    
    ray.init(ignore_reinit_error=True,include_dashboard=False)

    go_parent = None
    if go_type == "bp":
        go_parent = "GO:0008150"
    elif go_type == "cc":
        go_parent = "GO:0005575"
    elif go_type == "mf":
        go_parent = "GO:0003674"
    else:
        raise Exception(f"Unknown go type: {go_type}")

    go_parent_mapping = go_parent_mapping[go_parent_mapping["root_parent"] == go_parent]
    
    pbar = ProgressBar(len(protein_to_go_mapping),description=f"Generating go term mapping for : {go_type}")
    pbar_actor = pbar.actor
    go_parent_mapping_id = ray.put(go_parent_mapping)

    mapping_ray = []
    for item in protein_to_go_mapping.iterrows():
        result = _filter_terms_for_protein.remote(item,go_parent_mapping_id,pbar_actor)
        mapping_ray.append(result)

    pbar.print_until_done()
    mappings = ray.get(mapping_ray)

    mappings_pd = pd.DataFrame(data=mappings,columns=["protein","go_terms"])
    return mappings_pd

@ray.remote
def _filter_terms_for_protein(item,go_parent_mapping,pbar):
    row = item[1]
    protein = row["protein"]
    go_terms = row["go_terms"].split(",")
    filtered_terms = []
    for go_term in go_terms:
        filtered_go_parents = go_parent_mapping[go_parent_mapping["go_node"] == go_term]
        if len(filtered_go_parents) > 0:
            filtered_terms.append(go_term)

    mapping = (protein,",".join(filtered_terms))

    pbar.update.remote(1)

    return mapping

def _get_go_terms(protein_name,data_dir):
    file_name = f"{data_dir}/raw/uniprot/{protein_name}.xml"

    # check if file exists
    if path.exists(file_name):  
        uniprot_xml = open(file_name).read()
        
        # get all the go terms
        go_terms = re.findall(r'(?<=GO:)(.*)(?=")',uniprot_xml)
        
        # remove duplicates
        go_terms = list(set(go_terms))

        # add the GO: identifier to every term
        go_terms = list(map(lambda x : f"GO:{x}", go_terms))

        return go_terms
    
    # else
    else:
        logger.warning(f"NO uniprot file for : {protein_name}")
        # return an empty list
        return []

def generate_go_parent_mapping(data_dir):
    # load the obo file
    logger.info("Reading GO ontology file")
    go_graph = obonet.read_obo(f"{data_dir}/raw/misc/go_basic.obo")

    # convert to plain old digraph
    go_graph = nx.DiGraph(go_graph)

    # remove node attributes by serializing to adjacency matrix
    edgelist = nx.to_edgelist(go_graph)
    go_graph = nx.DiGraph(edgelist)

    # reverse the direction of edges so that root nodes have highest degree
    go_graph = go_graph.reverse()

    # find the parents of all nodes
    mappings = []
    for node in tqdm(go_graph.nodes()):
        parent = find_root(go_graph,node)
        mapping = (node,parent)
        mappings.append(mapping)

    # save to disk
    df = pd.DataFrame.from_records(mappings,columns=["go_node","root_parent"])
    return df

def find_root(G,node):
    if list(G.predecessors(node)):  #True if there is a predecessor, False otherwise
        root = find_root(G,list(G.predecessors(node))[0])
    else:
        root = node
    return root

def generate_go_subgraph(data_dir,term_type):
    
    go_parent = None
    if term_type == "bp":
        go_parent = "GO:0008150"
    elif term_type == "cc":
        go_parent = "GO:0005575"
    elif term_type == "mf":
        go_parent = "GO:0003674"

    if go_parent is None:
        raise Exception(f"Given term type not recognized: {term_type}")        

    # load go parent mapping
    go_parent_mapping_pd = pd.read_csv(f"{data_dir}/processed/go_nodes_parent_mapping.csv")

    # subset for the nodes having releavant parents
    go_parent_mapping_pd = go_parent_mapping_pd[go_parent_mapping_pd["root_parent"] == go_parent]
    target_nodes = go_parent_mapping_pd["go_node"].to_list()

    # load the obo file
    logger.info("Reading GO ontology file")
    go_graph = obonet.read_obo(f"{data_dir}/raw/misc/go_basic.obo")

    # convert to plain old digraph
    go_graph = nx.DiGraph(go_graph)

    # remove node attributes by serializing to adjacency matrix
    edgelist = nx.to_edgelist(go_graph)
    go_graph = nx.DiGraph(edgelist)

    # reverse the direction of edges so that root nodes have highest degree
    go_graph = go_graph.reverse()

    # sub graph to keep only target nodes
    go_sub_graph = go_graph.subgraph(target_nodes)
    logger.info(f"Number of nodes in {term_type}: {len(go_sub_graph.nodes())}")

    assert len(go_graph.nodes()) != len(go_sub_graph.nodes()) , "No nodes removed"
    
    return go_sub_graph


def generate_go_subgraph_with_edge_type(data_dir,term_type):
    
    go_parent = None
    if term_type == "bp":
        go_parent = "GO:0008150"
    elif term_type == "cc":
        go_parent = "GO:0005575"
    elif term_type == "mf":
        go_parent = "GO:0003674"

    if go_parent is None:
        raise Exception(f"Given term type not recognized: {term_type}")        

    # load go parent mapping
    go_parent_mapping_pd = pd.read_csv(f"{data_dir}/processed/go/go_nodes_parent_mapping.csv")

    # subset for the nodes having releavant parents
    go_parent_mapping_pd = go_parent_mapping_pd[go_parent_mapping_pd["root_parent"] == go_parent]
    target_nodes = go_parent_mapping_pd["go_node"].to_list()

    # load the obo file
    logger.info("Reading GO ontology file")
    go_graph = obonet.read_obo(f"{data_dir}/raw/misc/go_basic.obo")

    # sub graph to keep only target nodes
    go_sub_graph = go_graph.subgraph(target_nodes)
    logger.info(f"Number of nodes in {term_type}: {len(go_sub_graph.nodes())}")

    assert len(go_graph.nodes()) != len(go_sub_graph.nodes()) , "No nodes removed"

    go_sub_graph_clean = nx.MultiDiGraph()
    edges = list(go_sub_graph.edges())
    for edges in tqdm(edges,desc=f"Generating go subgraph for : {term_type}"):
        head = edges[0]
        tail = edges[1]
        relationship = list(go_sub_graph[head][tail].keys())[0]
        go_sub_graph_clean.add_edge(head,tail,edge_type=relationship)

        # add inverse
        if relationship == "is_a":
            go_sub_graph_clean.add_edge(tail,head,edge_type="has_a")

    return go_sub_graph_clean


def generate_region_mapping():
    # load the obo file
    logger.info("Reading GO ontology file")
    go_graph = obonet.read_obo(f"data/raw/misc/go.obo")

    # convert to plain old digraph
    go_graph = nx.DiGraph(go_graph)

    # remove node attributes by serializing to adjacency matrix
    edgelist = nx.to_edgelist(go_graph)
    go_graph = nx.DiGraph(edgelist)

    # reverse the direction of edges so that root nodes have highest degree
    go_graph = go_graph.reverse()

    # find the parents of all nodes
    mappings = []
    extracellular = "GO:0005576"
    intracellular = "GO:0005622"
    nucleus = "GO:0005634"
    cytoplasm = "GO:0005737"
    cc_go_terms_pd = _get_cellular_component_go_terms()
    cc_go_terms = cc_go_terms_pd["go_node"].tolist()

    for node in tqdm(cc_go_terms):
        
        is_extracellular = nx.has_path(go_graph,extracellular,node)
        is_intracelular = nx.has_path(go_graph,intracellular,node)
        is_oncell = (is_extracellular == is_intracelular == False)
        is_in_nucleus = nx.has_path(go_graph,nucleus,node)
        is_in_cytoplasm = nx.has_path(go_graph,cytoplasm,node)

        mappings.append((node,is_intracelular,is_extracellular,is_oncell,is_in_nucleus,is_in_cytoplasm))

    # save to disk
    df = pd.DataFrame.from_records(mappings,columns=["go_node","is_intracellular","is_extracellular","is_oncell","is_in_nucleus","is_in_cytoplasm"])
    return df

def _get_cellular_component_go_terms():
    # read the data
    go_terms_pd = pd.read_csv("data/processed/go/go_parent_mapping.csv")
    cc_go_terms_pd = go_terms_pd[go_terms_pd["root_parent"] == "GO:0005575"].copy()
    return cc_go_terms_pd

def generate_protein_region_mapping():

    # load go parent mapping
    go_parent_mapping_pd = pd.read_csv("data/processed/go/go_parent_mapping.csv")

    # load go region mapping
    go_region_mapping_pd = pd.read_csv("data/processed/go/go_region_mapping.csv")

    # load protein to go mapping
    protein_to_go_mapping_pd = pd.read_csv("data/processed/go/protein_to_go_mapping.csv")

    protein_region_mapping = []
    for _, row in tqdm(protein_to_go_mapping_pd.iterrows(),total=len(protein_to_go_mapping_pd)):
        protein = row["protein"]
        go_terms = row["go_terms"]
        if type(go_terms) is str:
            go_terms = go_terms.split(",")
        else:
            go_terms = [] 

        # filter terms for cellular component
        go_terms_cc,skipped_count = filter_for_cellular_component(go_terms,go_parent_mapping_pd)
        if skipped_count > 5:
            logger.warning(f"Too many go terms skipped for : {protein}")

        # determine overall cellular region for go_terms
        cc_region = get_region(go_terms_cc,go_region_mapping_pd)

        is_extracellular = (cc_region == GO_REGION.EXTRA_CELLULAR)
        is_nucleus = (cc_region == GO_REGION.NUCLEUS)
        is_cytoplasm  = (cc_region == GO_REGION.CYTOPLASM)

        protein_region_mapping.append((protein,is_extracellular,is_nucleus,is_cytoplasm))

    protein_region_mapping_pd = pd.DataFrame(data=protein_region_mapping,columns=["protein","is_extracellular","is_nucleus","is_cytoplasm"])

    return protein_region_mapping_pd


def filter_for_cellular_component(go_terms,go_parent_mapping):
        go_cellular = []
        parent_cc = "GO:0005575"
        skipped_count = 0
        for term in go_terms:
            filtered_df = go_parent_mapping[go_parent_mapping["go_node"] == term]
            if len(filtered_df) > 0:
                parent = filtered_df.iloc[0]["root_parent"]
                if parent == parent_cc:
                    go_cellular.append(term)
            else:
                skipped_count =+1

        return go_cellular,skipped_count


def get_region(go_terms,go_region_mapping):
        num_extra_cellular = 0
        num_nucleus = 0
        num_cytoplasm  = 0

        for go_term in go_terms:
            region_row = go_region_mapping[go_region_mapping["go_node"] == go_term].iloc[0]
            if region_row["is_extracellular"] == True:
                num_extra_cellular += 1
            if region_row["is_in_nucleus"] == True:
                num_nucleus += 1
            if region_row["is_in_cytoplasm"] == True:
                num_cytoplasm += 1

        if num_extra_cellular > num_nucleus and num_extra_cellular > num_cytoplasm:
            return GO_REGION.EXTRA_CELLULAR
        elif num_nucleus > num_extra_cellular and num_nucleus > num_cytoplasm:
            return GO_REGION.NUCLEUS
        else:
            return GO_REGION.CYTOPLASM


def create_labelled_subset(go_graph,node,num_nodes):
    logger.info(f"Creating labelled subset: {node}")
    desc = list(nx.nodes(nx.dfs_tree(go_graph, node,10)))
    pos_samples = desc[:num_nodes]

    all_nodes = list(go_graph.nodes())
    cand_neg_samples = list(set(all_nodes) - set(pos_samples))
    
    neg_samples = []
    for neg_sample in cand_neg_samples:
        if nx.has_path(go_graph,node,neg_sample) == False:
            neg_samples.append(neg_sample)

    neg_samples = neg_samples[:len(pos_samples)]

    # labels
    import numpy as np
    pos_labels = np.ones((len(pos_samples),))
    neg_labels = np.zeros_like(pos_labels)

    labels = np.concatenate((pos_labels,neg_labels),axis=0)

    labelled_subset_df = pd.DataFrame()
    labelled_subset_df["go_node"] = pos_samples + neg_samples
    labelled_subset_df["label"] = labels

    # shuffle
    labelled_subset_df = labelled_subset_df.sample(frac=1)

    return labelled_subset_df





