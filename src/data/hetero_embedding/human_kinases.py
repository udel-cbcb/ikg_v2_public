from pathlib import Path
import pandas as pd
from loguru import logger


def build():
    data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
    kinases_raw = pd.read_csv(f"{data_dir}/raw/iptmnet/human_kinases_raw.csv")

    # get the gene
    kinases_raw = kinases_raw["UniProt_ID"].dropna().tolist()
    logger.info(f"Number of human kinases raw: {len(kinases_raw)}")

    # load proteins
    proteins = pd.read_csv(f"{data_dir}/processed/proteins.csv")["protein"].tolist()

    # only keep kinases that are present in our proteins
    kinases_filtered = []
    for kinase in kinases_raw:
        if kinase in proteins:
            kinases_filtered.append(kinase)

    logger.info(f"Number of human kinases filtered: {len(kinases_filtered)}")

    # filter substrates
    substrates_filtered = []
    for substrate in proteins:
        if substrate not in kinases_filtered:
            substrates_filtered.append(substrate)

    # shuffle substrates
    from sklearn.utils import shuffle
    substrates_filtered = shuffle(substrates_filtered)

    # keep only equal to kinases
    substrates_filtered = substrates_filtered[0:len(kinases_filtered)]

    proteins_filtered = kinases_filtered + substrates_filtered
    proteins_filtered = shuffle(proteins_filtered)

    kinase_substrates = []
    for protein in proteins_filtered:
        if protein in kinases_filtered:
            kinase_substrates.append((protein,1))
        else:
            kinase_substrates.append((protein,0))

    # kinase_substrate df
    kinase_substrate_df = pd.DataFrame(data=kinase_substrates,columns=["proteins","is_kinase"])

    # save
    out_file = f"{data_dir}/processed/proteins/kinase_substrates.csv"
    logger.info(f"Saving output to: {out_file}")
    kinase_substrate_df.to_csv(out_file,index=False)
    logger.info("Done")


if __name__ == "__main__":
    build()