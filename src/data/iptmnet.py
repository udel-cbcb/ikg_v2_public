import src.data.data_utils as du
import networkx as nx
from loguru import logger
import pandas as pd
from tqdm import tqdm
import math
import src.data.constants as constants

def build_graph(data_dir,event=None):
    iptmnet_df = pd.read_csv(f"{data_dir}/raw/iptmnet/MV_EVENT_human_only.tsv",sep="\t")
    logger.info(f"Loaded data : {iptmnet_df.shape}")

    # extract entities
    substrates  = []
    kinases = []
    kg_graph = nx.DiGraph()
    for _index, row in tqdm(iptmnet_df.iterrows(),total=iptmnet_df.shape[0]):
        event_type = _extract_event_type(row)
        
        # if no event is provided or if event type is equal to provided event type
        if event == None or event == event_type.lower():
            
            # get the entities
            substrate, kinase = extract_entities(row)
            substrates.append(substrate)
            kinases.append(kinase)

            # add new edge to graph for event type
            if(kg_graph.has_edge(kinase,substrate)):
                # get edge type
                existing_edge_type = kg_graph.get_edge_data(kinase,substrate)
                
                # if existing edge is phopsph edge
                if existing_edge_type == "p":
                    # then skip this
                    pass
                else:
                    kg_graph.add_edge(kinase,substrate,edge_type=event_type)
            else:
                kg_graph.add_edge(kinase,substrate,edge_type=event_type)

    # combine both the list together
    proteins = []
    proteins.extend(substrates)
    proteins.extend(kinases)

    # remove duplicates
    proteins = list(set(proteins))

    logger.info(f"Total number of unique proteins : {len(proteins)}")
    logger.info(f"Total number of connections: {len(kg_graph.edges)}")
    return iptmnet_df, kg_graph, proteins

def _extract_event_type(row):
    return row["event_label"]

def extract_entities(row):
    substrate = extract_substrate(row)
    kinase = extract_kinase(row)

    return substrate, kinase

def extract_substrate(data_row):
    # check the entity type
    sub_type = data_row["sub_type"]
    
    substrate = ""

    # if entity type is pro id
    if sub_type == "pro_id":
        substrate = data_row["sub_xref"]
        if isinstance(substrate,str) == False and math.isnan(substrate):
            substrate = data_row["SUB_CODE"] 
            
    # if entity type is uniprot ac
    elif sub_type == "uniprot_ac":
        substrate = data_row["SUB_CODE"]

    # else
    else:
        substrate = data_row["sub_form_code"]
    
    substrate_cleaned = clean_entity(substrate)

    # if cleaned entity is all number, then it is a pro_id
    if substrate_cleaned.isnumeric():
        substrate = substrate
    # else keep the cleaned entity
    else:
        substrate = substrate_cleaned

    return substrate

def extract_kinase(data_row):
    # check the entity type
    sub_type = data_row["sub_type"]
    
    kinase = ""

    # if entity type is pro id
    if sub_type == "pro_id":
        kinase = data_row["enz_xref"]
        if isinstance(kinase,str) == False and math.isnan(kinase):
            kinase = data_row["ENZ_CODE"] 
            
    # if entity type is uniprot ac
    elif sub_type == "uniprot_ac":
        kinase = data_row["ENZ_CODE"]

    # else
    else:
        kinase = data_row["enz_form_code"]
    
    kinase_cleaned = clean_entity(kinase)

    # if cleaned entity is all number, then it is a pro_id
    if kinase_cleaned.isnumeric():
        kinase = kinase
    # else keep the cleaned entity
    else:
        kinase = kinase_cleaned

    return kinase

def clean_entity(proteoform):
    proteoform_cleaned = proteoform.replace("PR:","")
    proteoform_cleaned = proteoform_cleaned.split("-")[0]
    
    return proteoform_cleaned
