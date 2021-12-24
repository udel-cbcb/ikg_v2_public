import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

def generate():
    # load file
    data_dir = f"{Path.home()}/data/ikg_v2_data/bioinformatics_analysis"
    idg_list = pd.read_excel(f"{data_dir}/raw/IDG_List_20210120.xlsx") 

    # keep only the kinases
    idg_kinases_pd = idg_list[idg_list["Family"] == "Kinase"]
    idg_kinases_pd = idg_kinases_pd[["Gene_2020"]]
    idg_kinases_pd = idg_kinases_pd.rename(columns={"Gene_2020": "kinase"})


    # save the list
    out_path = f"{data_dir}/processed/idg_kinases.csv"
    logger.info(f"Saving kinases to : {out_path}")
    idg_kinases_pd.to_csv(out_path,index=False)

    


if __name__ == "__main__":
    generate()