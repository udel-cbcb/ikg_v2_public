import pandas as pd
import networkx as nx
from tqdm import tqdm
from pyvis.network import Network
from loguru import logger

def create_viz_subset():
    # load the data
    ikg_pd = pd.read_csv("data/processed/kg/ikg.csv")
    ikg_pd = ikg_pd[ikg_pd["relation"] != "type_of"]

    # load parent mapping
    parent_mapping_pd = pd.read_csv("data/processed/go/go_parent_mapping.csv")
    
    # create graph
    logger.info("Creating graph")
    graph = nx.DiGraph()
    for _, row in tqdm(ikg_pd.iterrows(),total=len(ikg_pd)):
        head = row["head"]
        tail = row["tail"]
        rel = row["relation"]

        graph.add_edge(head,tail,type=rel)

    # get the subgraph
    logger.info("Finding descendents")
    desc = list(nx.descendants(graph,"Q8IV63"))
    desc.append("Q8IV63")
    subgraph = nx.subgraph(graph,desc)

    # viz
    logger.info("Creating viz network")
    net = Network("800px","800px")
    for edge in tqdm(subgraph.edges(data=True),total=len(subgraph.edges())):
        head = edge[0]
        tail = edge[1]

        net.add_node(head,color=get_color(head,parent_mapping_pd))
        net.add_node(tail,color=get_color(tail,parent_mapping_pd))
        net.add_edge(head,tail)

    net.show_buttons(filter_=['physics'])
    net.show("net.html")


def get_color(node,parent_mapping):
    if "GO:" in node:
        parent = get_parent(node,parent_mapping)
        if parent == "GO:0008150":
            return "green"
        elif parent == "GO:0003674":
            return "blue"
        else:
            return "purple"
    else:
        if node == "go_term":
            return "gray"
        elif node == "protein":
            return "pink"
        else:
            return "red"

def get_category(node,parent_mapping):
    if "GO:" in node:
        parent = get_parent(node,parent_mapping)
        if parent == "GO:0008150":
            return "bp"
        elif parent == "GO:0003674":
            return "mp"
        else:
            return "cc"
    else:
        if node == "go_term" or node == "protein":
            return "term_type"
        else:
            return "protein"

def get_edge_color(edge_name):
    if "phosp" in edge_name:
        return "red"
    if "annot" in edge_name:
        return "gray"

def get_parent(node, parent_mapping):
    parent = parent_mapping[parent_mapping["go_node"]==node]
    if len(parent) > 0:
        return parent.iloc[0]["root_parent"]
    else:
        return ""

    