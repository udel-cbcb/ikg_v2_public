from numpy import exp
import pandas as pd
from loguru import logger
from tqdm import tqdm
import math
import re
from os import path
import networkx as nx
import src.misc.utils as utils
from sklearn.model_selection import train_test_split
import numpy as np
import src.data.constants as constants
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing
import random
from sklearn.utils import shuffle
import torch

def get_pro_vector(protein_name):
    pass


def read_protein_names():
    proteins_txt = open("data/input/ml_data/iptmnet_network_proteins.txt")
    proteins = proteins_txt.read().splitlines()
    return proteins


def download_from_uniprot():
    protein_names = read_protein_names()
    for _protein in protein_names:
        pass


def get_go_terms(protein_name):

    file_name = f"data/input/uniprot/{protein_name}.xml"

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

        # return an empty list
        return []

def get_ec_category(protein_name):

    file_name = f"data/input/uniprot/{protein_name}.xml"

    # check if file exists
    if path.exists(file_name):  
        uniprot_xml = open(file_name).read()
        
        # get all the go terms
        ec_terms = re.findall(r'(?<=dbReference type="EC" id=")(.*)(?=")',uniprot_xml)
        
        # remove duplicates
        ec_terms = list(set(ec_terms))

        # keep only the first
        ec_category = "-1"
        if len(ec_terms) > 0:
            ec_term = ec_terms[0]
            ec_category = ec_term.split(".")[0]

        return ec_category
    
    # else
    else:

        # return an empty list
        return "NA"

def get_experimental_triples(data_dir):
    # load the experimentally validated relationships
    experimentally_validated = pd.read_csv(f"{data_dir}/raw/iptmnet/dark_kinase_substrates.txt",sep="\t")
    experimentally_validated = experimentally_validated[["KinaseAC","SubstrateAC"]]

    # clean the experimental data
    kinases = experimentally_validated["KinaseAC"] .apply(utils.clean_proteins).tolist()
    substrates = experimentally_validated["SubstrateAC"] .apply(utils.clean_proteins).tolist()

    # get edge tuples
    experimental_edges = [(kinases[i], "p", substrates[i]) for i in range(0, len(kinases))]

    experimental_edges_pd = pd.DataFrame(data=experimental_edges,columns=["head","relation","tail"])

    return experimental_edges_pd


def get_negative_edges(graph,num_negatives=10,seed=10):
    logger.info("Sampling negative edges")
    positive_edges = list(graph.edges())
    positive_edges_df = pd.DataFrame(data=positive_edges,columns=["head","tail"])
    corrupted_edges = []
    seed_index = seed
    for positive_edge in tqdm(positive_edges):

        # get all the edges where the head is not the given protein
        negative_edges_df = positive_edges_df[positive_edges_df["head"] != positive_edge[0]]
        
        # randomly choose an edge from the possible negative edges
        negative_edge = negative_edges_df.sample(n=num_negatives,random_state=seed_index)

        if len(negative_edge) == num_negatives:
            for index in range(0,num_negatives):
                negative_tail = negative_edge.iloc[index]["tail"]
                key=f"{positive_edge[0]}:{positive_edge[1]}"
                corrupted_edge = (key,positive_edge[0],negative_tail)
                corrupted_edges.append(corrupted_edge)
        else:
            logger.warning(f"Number of negative edges for : {positive_edge[0]} is not {num_negatives}. Please check your data. Quitting")

        # increment seed index so next time we have a different negative sample
        seed_index = seed_index + 7

    corrupted_edges_df = pd.DataFrame(data=corrupted_edges,columns=["key","head","tail"])
    return corrupted_edges_df

def create_train_val_test_splits(ikg_df,experimental_relations_df):
    
    # subset ikg_df to not include experimental relations
    logger.info(f"Before removing exp relation: {len(ikg_df)}")
    ikg_without_exp = ikg_df[~ikg_df.index.isin(experimental_relations_df.index)]
    logger.info(f"After removing exp relation: {len(ikg_without_exp)}")

    # get all non phosph relations
    ikg_non_phosp = ikg_without_exp[ikg_without_exp["relation"] != "p"]
    ikg_only_phosp = ikg_without_exp[ikg_without_exp["relation"] == "p"]

    triples = list(ikg_only_phosp.to_records(index=False))
    
    # create train test split
    triples = shuffle(triples,random_state=10)
    train_triples, test_triples = train_test_split(triples,random_state=20,shuffle=True,test_size=0.20)
        
    # create val split
    train_triples, val_triples = train_test_split(train_triples,random_state=20,shuffle=True,test_size=0.20)

    # non phosph to train
    

    # add experimental triples to test set
    triples_exp = list(experimental_relations_df.to_records(index=False))
    test_triples.extend(triples_exp)

    return train_triples, val_triples, test_triples


def to_indexed(triples,entity_mapping,relation_mapping):
    
    # get the index for every triple
    triples_indexed = []
    for triple in triples:
        head = entity_mapping[triple[0]]
        relation = relation_mapping[triple[1]]
        tail = entity_mapping[triple[2]]
        triples_indexed.append((head,relation,tail))

    # to numpy
    triples_indexed = np.asarray(triples_indexed)

    return triples_indexed


def create_entity_mapping(entity_mapping_pd):
    entity_mapping = {}
    for _,row in entity_mapping_pd.iterrows():
        entity = row["entity"]
        index = row["index"]
        entity_mapping[entity] = index

    logger.info(f"Number of entities: {len(entity_mapping)}")

    return entity_mapping

def create_relation_mapping(relation_mapping_pd):
    relation_mapping = {}
    for _,row in relation_mapping_pd.iterrows():
        entity = row["relation"]
        index = row["index"]
        relation_mapping[entity] = index

    return relation_mapping


def filter_for_phosp(ikg_pd):
    # keep only phosphorylation relations
    ikg_pd_filtered = ikg_pd[ikg_pd["relation"] == constants.PHOSPHORYLATES]
    return ikg_pd_filtered

def get_negatives(triples, negative_triples_df,num_negatives):
    negative_triples =  []
    for pos_triple in tqdm(triples):
        negative_items = negative_triples_df.query(f"key == '{pos_triple[0]}:{pos_triple[2]}'")
        for index in range(0,num_negatives):
            negative_item = negative_items.iloc[index]
            negative_triple =(negative_item[1],constants.PHOSPHORYLATES,negative_item[2])
            negative_triples.append(negative_triple)

    return negative_triples


def get_edge_embeddings(triples, embeddings_pd):
    embeddings = []
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(processes=cpu_count-1)
    do_work = lambda edge: get_embedding_vector_for_edge(edge,embeddings_pd)
    embeddings = pool.map(do_work, triples)
    embeddings_vec = np.concatenate(embeddings,axis=0)
    return embeddings_vec


def get_embedding_vector_for_edge(edge,embeddings_pd):
    source_node = edge[0]
    target_node = edge[2]
    shape = embeddings_pd.shape
    source_embedding = embeddings_pd[embeddings_pd["entity"] == source_node].drop(columns=["entity"]).to_numpy()
    target_embedding = embeddings_pd[embeddings_pd["entity"] == target_node].drop(columns=["entity"]).to_numpy()
    
    # if we get empty vectors then zero arrs
    if source_embedding.size == 0:
        logger.warning(f"Embedding is null for {source_node}")
        source_embedding = np.zeros((1,shape[1]-1))

    if target_embedding.size == 0:
        logger.warning(f"Embedding is null for {target_node}")
        target_embedding = np.zeros((1,shape[1]-1))

    embedding_np = operator_hadamard(source_embedding,target_embedding)
    
    return embedding_np

def operator_hadamard(u, v):
    return u * v

def operator_cat(u, v):
    return np.concatenate([u,v],axis=1)

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
    tail_node = random_edge[2]

    # create a new corrupted edge
    negative_edge = (edge[0],constants.PHOSPHORYLATES, tail_node)

    return negative_edge

def to_edges(triples):
    edges = []
    for triple in triples:
        edges.append((triple[0],triple[2]))

    return edges


def create_negatives(pos_edges,num_nodes):
    pos_edges_list = torch.clone(pos_edges).tolist()
    num_edges = len(pos_edges_list)

    # corrupt the tail
    neg_edges = []
    for i in tqdm(range(num_edges),total=num_edges):
        head = pos_edges_list[i][0]
        tail_indices = torch.randint(high=num_edges,size=(100,))
        num_added = 0
        for tail_idx in tail_indices:
            tail = int(pos_edges_list[tail_idx][1])
            edge = (head,tail)
            
            if edge not in pos_edges_list:
                neg_edges.append(edge)
                num_added = num_added + 1
                if num_added == 1:
                    break

    neg_edges_tensor = torch.Tensor(neg_edges)

    return neg_edges_tensor

def discretize(x):
    if x >= 0.5:
        return 1
    else:
        return 0

def filter_for_dataframe(target_df,filter_df):
    filtered_tuples = []
    target_tuples = target_df.to_records(index=False)
    filter_tuples = filter_df.to_records(index=False)

    for record in tqdm(target_tuples):
        if record not in filter_tuples:
            filtered_tuples.append(record)

    filtered_df = pd.DataFrame.from_records(data=filtered_tuples,columns=target_df.columns)
    return filtered_df