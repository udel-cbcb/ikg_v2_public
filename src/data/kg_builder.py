from math import gamma
import src.data.iptmnet as iptmnet
from loguru import logger
import pandas as pd
import src.data.go as go
import src.data.pfam as pfam
import src.data.pro as pro
from tqdm import tqdm
import src.data.data_utils as du
import networkx as nx
import ray
from src.misc.pbar import ProgressBar

def build_kg():
    
    # read the data
    _, graph, proteins, protein_graph = iptmnet.build_graph(event="p")

    # create negative samples for phosphorylation edges
    negative_edges_df = du.get_negative_edges(protein_graph,seed=10)    
    negative_edges_df.to_csv("data/processed/kg/negative_edges.csv",index=False)

    # experimental relations
    experimental_relations_df = du.get_experimental_triples()
    experimental_relations_df.to_csv("data/processed/kg/experimental_triples.csv",index=False)
    
    # annotate proteins with go terms
    graph = go.annotate_with_go_terms(graph,proteins)

    # merge with go graph
    graph = go.merge_with_go_graph(graph)
    
    # annotate with pfam
    #graph = pfam.annotate_with_pfam(graph,proteins)

    # annotate with pro
    #graph = pro.merge_with_pro(graph,proteins)

    # stats
    logger.info(f"Total number of entities: {len(graph.nodes())}")
    logger.info(f"Total number of relationships: {len(graph.edges())}")

    # save proteins list
    proteins_pd = pd.DataFrame(data=proteins,columns=["protein"])
    proteins_pd.to_csv("data/processed/kg/proteins.csv",index=False)

    # serialize    
    serialize(graph,"data/processed/kg/ikg.csv")

    # serialize to indexed triples
    serialize_indexed(graph,"data/processed/kg")

    # to GML
    nx.write_gml(graph,"data/processed/kg/ikg.gml")

    # generate go term mapping
    logger.info("Generating go term mapping")
    go_term_mapping_pd = go.generate_term_mapping(proteins)
    go_term_mapping_pd.to_csv("data/processed/go/protein_to_go_mapping.csv",index=False)

    # generate go parent mapping
    logger.info(f"Generate go parent mapping")
    go_parent_mapping_pd = go.generate_go_parent_mapping()
    go_parent_mapping_pd.to_csv("data/processed/go/go_parent_mapping.csv",index=False)

    # generate go cellular region mapping
    logger.info(f"Generate go region mapping")
    go_region_mapping_pd = go.generate_region_mapping()
    go_region_mapping_pd.to_csv("data/processed/go/go_region_mapping.csv",index=False)


def serialize(graph,out_file):

    # convert graph to triples
    triples_pd = graph_to_triples(graph)

    logger.info(f"Serializing to : {out_file}")
    triples_pd.to_csv(out_file,index=False)

def serialize_indexed(graph,out_path,file_prefix):

    # loop over all the edges of the graph
    logger.info("Serializing graph indexed - Getting all triples")
   
    # convert graph to triples
    triples_pd = graph_to_triples(graph)

    # get all the unique entities
    heads = triples_pd["head"].unique().tolist()
    tails = triples_pd["tail"].unique().tolist()
    
    # merge both the list
    entities = []
    entities.extend(heads)
    entities.extend(tails)
    entities = list(set(entities))

    # get relationships
    relationships = triples_pd["relation"].unique().tolist()

    # create a mapping of entities
    entity_map = {}
    logger.info("Serializing graph indexed - creating entity mapping")
    for index,entity in tqdm(enumerate(entities),total=len(entities)):
        entity_map[entity] = index
    
    # create a mapping of relationships
    relations_map = {}
    num_entities = len(entity_map)
    logger.info("Serializing graph indexed - creating relation mapping")
    for index, relation in tqdm(enumerate(relationships),total=len(relationships)):
        relations_map[relation] = num_entities+index

    triples_indexed_result = []
    triples_indexed = []
    entity_map_id = ray.put(entity_map)
    relations_map_id = ray.put(relations_map)
    triples_records = triples_pd.to_records(index=False)

    logger.info("Serializing graph indexed - creating work queue")
    pbar = ProgressBar(len(triples_pd),description=f"Generating indexed triples")
    pbar_actor = pbar.actor
    for index, record in enumerate(triples_records):
        result = _to_indexed_tuples.remote(record,entity_map_id,relations_map_id,pbar_actor)
        triples_indexed_result.append(result)

        if index % 2000 == 0:
            pbar.update()

    pbar.print_until_done()
    triples_indexed = ray.get(triples_indexed_result)

    # convert to dataframe
    triples_pd_indexed = pd.DataFrame(data=triples_indexed,columns=["head","relation","tail"])

    # save to csv
    logger.info(f"Serializing indexed tuples to : {out_path}")
    triples_pd_indexed.to_csv(f"{out_path}/{file_prefix}_triples_indexed.csv",index=False)

    # serialize entity mapping
    entity_map_pd = pd.DataFrame(data=list(entity_map.items()),columns=["entity","index"])
    entity_map_pd.to_csv(f"{out_path}/{file_prefix}_entity_map.csv",index=False)

    # serialize entity mapping
    relations_map_pd = pd.DataFrame(data=list(relations_map.items()),columns=["relation","index"])
    relations_map_pd.to_csv(f"{out_path}/{file_prefix}_relation_map.csv",index=False)       

@ray.remote
def _to_indexed_tuples(record,entity_map,relations_map,pbar_actor):
    
    # get labels
    head = record[0]
    relation = record[1]
    tail = record[2]

    # get indexes
    head_index = entity_map[head]
    relation_index = relations_map[relation]
    tail_index = entity_map[tail]

    pbar_actor.update.remote(1)
    
    return (head_index,relation_index,tail_index)

def graph_to_triples(graph):
    def _to_triples_item(edge):
        head = edge[0]
        relation = edge[2]["edge_type"]
        tail = edge[1]
        triple = (head,relation,tail)
        return triple
        

    triples_result = []
    triples = []
    edges = list(graph.edges(data=True))

    logger.info("Converting graph to triples")
    for edge in tqdm(edges,leave=True):
        result = _to_triples_item(edge)
        triples_result.append(result)
    
    triples = triples_result

    triples_pd = pd.DataFrame.from_records(triples,columns=["head","relation","tail"])
    return triples_pd