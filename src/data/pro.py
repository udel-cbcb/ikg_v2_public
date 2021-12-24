from os import path
import re
import copy
from tqdm import tqdm
from loguru import logger
import obonet
import src.data.iptmnet as iptmnet

def merge_with_pro(graph,proteins):
    # make a copy
    graph_copy = copy.deepcopy(graph)

    logger.info("Loading pro ontology")
    pro_graph = obonet.read_obo("data/raw/misc/pro_reasoned.obo")

    # loop over all nodes
    logger.info("Merging with PRO")
    pro_nodes = list(pro_graph.nodes(data=True))

    for node in tqdm(pro_nodes):
        node_id = node[0]

        # clean the node id
        node_id_cleaned = iptmnet.clean_entity(node_id)

        # check if it exists in proteins
        if node_id_cleaned in proteins:
            # then add a new is_a relationship
            graph_copy.add_edge(node_id_cleaned,node_id,type="is_exactly")

            # add type of node_id
            if "namespace" in node[1]:
                node_id_type = node[1]["namespace"]
            else:
                node_id_type = node[1]["comment"]               

            graph_copy.add_edge(node_id,node_id_type,type="type_of")

        # add is_a relations
        if "is_a" in node[1]:
            is_a_nodes = node[1]["is_a"]
            for is_a_node in is_a_nodes:
                graph_copy.add_edge(node_id,is_a_node,type="is_a")

                # find out type of node
                graph_copy.add_edge(is_a_node,"pro_term",type="type_of")

    return graph_copy



