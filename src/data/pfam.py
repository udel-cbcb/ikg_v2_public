from os import path
from loguru import logger
import copy
from tqdm import tqdm
import re

def annotate_with_pfam(graph,proteins):
    logger.info("Annotating with PFAM")

    # make a copy
    graph_copy = copy.deepcopy(graph)

    for protein in tqdm(proteins):
        pfam_ids = _get_pfam_family(protein)

        for pfam_id in pfam_ids:
            graph_copy.add_edge(protein,pfam_id,type="belongs_to_family")

            # type of edge
            graph_copy.add_edge(pfam_id,"go_term",type="type_of")

    
    return graph_copy


def _get_pfam_family(protein_name):
    file_name = f"data/raw/uniprot/{protein_name}.xml"

    # check if file exists
    if path.exists(file_name):  
        uniprot_xml = open(file_name).read()
        
        # get all the go terms
        pfam_ids = re.findall(r'(?<=dbReference type="Pfam" id=")(.*)(?=")',uniprot_xml)
        
        # remove duplicates
        pfam_ids = list(set(pfam_ids))

        return pfam_ids
    
    # else
    else:

        # return an empty list
        return []