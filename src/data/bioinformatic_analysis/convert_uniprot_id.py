import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

"""
****** 
BEFORE RUNNING THIS SCRIPT RUN THE generate_kinase_list.py script.
Use the idg_kinases.csv file to get a list of uniprot ids
******
"""
def convert():
    data_dir = f"{Path.home()}/data/ikg_v2_data/bioinformatics_analysis"

    # load kinases uniprot
    kinases_uniprot = pd.read_csv(f"{data_dir}/raw/kinases_uniprot.tsv",sep="\t")
    kinases_uniprot = kinases_uniprot[["Entry"]]

    idg_kinases_pd = kinases_uniprot.rename(columns={"Entry":  "kinase"})

    # load proteins
    proteins_pd = pd.read_csv(f"{data_dir}/processed/proteins.csv")

    # to list
    logger.info("Converting to list")
    idg_kinases = idg_kinases_pd["kinase"].to_list()
    proteins = proteins_pd["protein"].to_list()

    # get target kinases
    target_kinases = []
    for kinase in tqdm(idg_kinases):
        if kinase in proteins:
            target_kinases.append(kinase)

    target_kinases_pd = pd.DataFrame(data=target_kinases,columns=["kinase"])
    
    # save
    out_path = f"{data_dir}/processed/target_kinases.csv"
    logger.info(f"Saving kinases to : {out_path}")
    target_kinases_pd.to_csv(out_path,index=False)

if __name__ == "__main__":
    convert()