from operator import pos
import random
from tqdm import tqdm
import networkx as nx
from loguru import logger

def sample_negative_edges(edges,num_negative=1, seed=10,show_progress=True):
    random.seed(seed)
    positive_edges = edges
    negative_edges = []
    for edge in tqdm(positive_edges,total=len(positive_edges),disable=not show_progress):
        
        for _ in range(0,num_negative):
            
            # create negative edge
            negative_edge = _create_corrupted_edge(positive_edges,edge)
                        
            # check if this edge exists already
            while negative_edge in positive_edges:
                negative_edge = _create_corrupted_edge(positive_edges,edge)

            # append to negative edges
            negative_edges.append(negative_edge)

    return negative_edges


def _create_corrupted_edge(positive_edges,edge):
    # radomly pick an egde from positive edges
    random_edge = random.sample(positive_edges,1)[0]

    # get the tail node
    tail_node = random_edge[1]

    # create a new corrupted edge
    negative_edge = (edge[0], tail_node)

    return negative_edge


def triples_to_graph(triples_df):
    graph = nx.DiGraph()
    logger.info("Converting triples to graph")
    for _, row in tqdm(triples_df.iterrows(),total=len(triples_df)):
        head = row["head"]
        edge_type = row["relation"]
        tail = row["tail"]
        graph.add_edge(head,tail,edge_type=edge_type)

    return graph