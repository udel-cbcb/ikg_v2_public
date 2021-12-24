import pandas as pd
from pathlib import Path
from loguru import logger

def generate():
   data_dir = f"{Path.home()}/data/ikg_v2_data/bioinformatics_analysis"

   # load all the edges
   train_edges = pd.read_csv(f"{data_dir}/processed/edges/positive_edges_train.csv")
   test_edges = pd.read_csv(f"{data_dir}/processed/edges/positive_edges_test.csv")
   val_edges = pd.read_csv(f"{data_dir}/processed/edges/positive_edges_train.csv")

   all_edges = pd.concat([train_edges,test_edges,val_edges])

   # load idg kinases
   idg_kinases = pd.read_csv(f"{data_dir}/processed/target_kinases.csv")["kinase"].to_list() 

   # filter for kinases that have no or very little information in iptmnet
   iptmnet_understudied_kinases = []
   for kinase in idg_kinases:
       num_substrate = len(all_edges[all_edges["head"] == kinase])
       num_kinases = len(all_edges[all_edges["tail"] == kinase])
       #print(f"{kinase}:{num_substrate}")
       if (num_kinases <=2) or (num_substrate <= 2):
           iptmnet_understudied_kinases.append(kinase)

   logger.info(f"Number of understudied kinases with respect to iPTMnet: {len(iptmnet_understudied_kinases)}")

   # load predictions 
   prediction_pd = pd.read_csv(f"{data_dir}/results/predicted_edges.csv")
   logger.info(f"Total number of predictions: {len(prediction_pd)}") 

   # keep only high confidence predictions
   hq_predictions_pd = prediction_pd[prediction_pd["proba"] >= 0.95]
   logger.info(f"Total number of HQ predictions: {len(hq_predictions_pd)}") 

   # filter HQ predictions for these kinases
   hq_predictions_pd_iptmnet = hq_predictions_pd[hq_predictions_pd["kinase"].isin(iptmnet_understudied_kinases)]
   hq_predictions_pd_iptmnet_grouped = hq_predictions_pd_iptmnet.groupby("kinase").count().reset_index().sort_values(by=["proba"],ascending=False) 

   print(hq_predictions_pd_iptmnet_grouped)

   hq_predictions_pd_iptmnet_grouped.to_csv(f"{data_dir}/results/hq_substrate_count.csv",index=False)

   for _, row in hq_predictions_pd_iptmnet_grouped.iterrows():
       kinase = row["kinase"]
       substrates = hq_predictions_pd[hq_predictions_pd["kinase"] == kinase]["substrate"].to_list()
       substrates_pd = pd.DataFrame(data=substrates,columns=["substrates"])
       out_path = f"{data_dir}/results/substrates/{kinase}.csv"
       logger.info(f"Saving the substrates to : {out_path}")  
       substrates_pd.to_csv(out_path,index=False)

    


if __name__ == "__main__":
    generate()