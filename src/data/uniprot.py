from os import path
import re
from tqdm import tqdm
import pandas as pd
from loguru import logger

def download_from_uniprot(protein_names):
    pass


def get_ec_mapping(proteins,data_dir):
    ec_mapping = []
    for protein in tqdm(proteins):
        ec_category = _get_ec_category(protein,data_dir)
        if ec_category != "" and ec_category != "-1":
            ec_mapping.append((protein,ec_category))

    ec_mapping_df = pd.DataFrame(data=ec_mapping,columns=["protein","ec_category"])
    logger.info(f"Number of proteins mapped to EC category: {len(ec_mapping_df)}")
    return ec_mapping_df

def _get_ec_category(protein_name,data_dir):

    file_name = f"{data_dir}/raw/uniprot/{protein_name}.xml"

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
        return ""

def get_ec_mapping_full(proteins,data_dir):
    ec_mapping = []
    for protein in tqdm(proteins):
        ec_category = _get_ec_category_full(protein,data_dir)
        if ec_category != "" and ec_category != "-1":
            ec_mapping.append((protein,ec_category))

    ec_mapping_df = pd.DataFrame(data=ec_mapping,columns=["protein","ec_category"])
    logger.info(f"Number of proteins mapped to EC category: {len(ec_mapping_df)}")
    return ec_mapping_df

def _get_ec_category_full(protein_name,data_dir):

    file_name = f"{data_dir}/raw/uniprot/{protein_name}.xml"

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
            if "2.7" in ec_term:
                ec_category = ".".join(ec_term.split(".")[0:3])

        return ec_category
    
    # else
    else:

        # return an empty list
        return ""

def create_ec_labels(ec_mapping_pd):

    ec_labels = []
    for index, row in tqdm(ec_mapping_pd.iterrows(),total=len(ec_mapping_pd)):
        protein = row["protein"]
        ec_category = row["ec_category"]
        label = 0
        if ec_category == "2":
            label = 1

        ec_labels.append((protein,label))

    ec_labels_pd = pd.DataFrame(data=ec_labels,columns=["protein","ec_label"])

    # balance out the labels
    ec_labels_pd_0 = ec_labels_pd[ec_labels_pd["ec_label"] == 0]
    ec_labels_pd_1 = ec_labels_pd[ec_labels_pd["ec_label"] == 1]

    num_0 = len(ec_labels_pd_0)

    ec_labels_pd_1_sampled = ec_labels_pd_1.sample(n=num_0,random_state=20)
   
    ec_labels_pd = pd.concat([ec_labels_pd_0,ec_labels_pd_1_sampled])
    ec_labels_pd = ec_labels_pd.sample(frac=1.0,random_state=20)

    return ec_labels_pd

def create_ec_labels_full(ec_mapping_pd):

    ec_labels = []
    for index, row in tqdm(ec_mapping_pd.iterrows(),total=len(ec_mapping_pd)):
        protein = row["protein"]
        ec_category = row["ec_category"]
        label = 0
        if ec_category == "2.7.11":
            label = 1

        ec_labels.append((protein,label))

    ec_labels_pd = pd.DataFrame(data=ec_labels,columns=["protein","ec_label"])

    # balance out the labels
    ec_labels_pd_0 = ec_labels_pd[ec_labels_pd["ec_label"] == 0]
    ec_labels_pd_1 = ec_labels_pd[ec_labels_pd["ec_label"] == 1]

    num_0 = len(ec_labels_pd_0)

    ec_labels_pd_1_sampled = ec_labels_pd_1.sample(n=num_0,random_state=20)
   
    ec_labels_pd = pd.concat([ec_labels_pd_0,ec_labels_pd_1_sampled])
    ec_labels_pd = ec_labels_pd.sample(frac=1.0,random_state=20)

    return ec_labels_pd