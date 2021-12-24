from pathlib import Path
import pandas as pd
import math
from loguru import logger

def build():
    data_dir = f"{Path.home()}/data/ikg_v2_data/feature_analysis"

    # load CTD mapping
    ctd_diseases = pd.read_csv(f"{data_dir}/raw/misc/CTD_diseases_wo_comments.csv")

    # keep only DiseaseID and AltID columns
    ctd_diseases = ctd_diseases[["DiseaseID","AltDiseaseIDs"]]

    # keep only doids in alt disease ids
    def keep_doids(row):
        # split
        if type(row) != float:
            alt_ids = row.split("|")
            for alt_id in alt_ids:
                if "DO" in alt_id:
                    doid = alt_id.replace("DO:","")
                    return doid

            return ""
        else:
            return ""

    
    ctd_diseases["AltDiseaseIDs"] = ctd_diseases["AltDiseaseIDs"].apply(keep_doids)

    # replace MESH:
    ctd_diseases["DiseaseID"] = ctd_diseases["DiseaseID"].apply(lambda x: x.replace("MESH:",""))

    # save
    out_path = f"{data_dir}/processed/misc/mesh_to_doid.csv"
    logger.info(f"Saving to : {out_path}")
    ctd_diseases.to_csv(out_path,index=False)

if __name__ == "__main__":
    build()