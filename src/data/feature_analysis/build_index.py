import pandas as pd
from pathlib import Path
from loguru import logger
from torch.utils import data
from tqdm import tqdm
from ikg_native import to_indexed_triples
import numpy as np

def build_index():
    data_dir = f"{Path.home()}/data/ikg_v2_data/feature_analysis"

    folds = 10

    features = ["go_bp","go_mf","pro","biokg_pathways","biokg_disease","biokg_disease_ontology","biokg_complex"]

    for fold_n in tqdm(range(folds),total=folds):

        for feature in features:

            train_triples_pd = pd.read_csv(f"{data_dir}/processed/annotated/{feature}/folds/fold_{fold_n}/train_triples.csv")
            val_triples_pd = pd.read_csv(f"{data_dir}/processed/folds/fold_{fold_n}/val_triples.csv")
            test_triples_pd = pd.read_csv(f"{data_dir}/processed/folds/fold_{fold_n}/test_triples.csv")
                
            # get all the heads
            entities = []
            entities.extend(train_triples_pd["head"].unique().tolist())
            entities.extend(val_triples_pd["head"].unique().tolist())
            entities.extend(test_triples_pd["head"].unique().tolist())

            # get all the tails
            entities.extend(train_triples_pd["tail"].unique().tolist())
            entities.extend(val_triples_pd["tail"].unique().tolist())
            entities.extend(test_triples_pd["tail"].unique().tolist())

            # merge and get unique
            entities = list(set(entities))

            # get relationships
            relationships = []
            relationships.extend(train_triples_pd["relation"].unique().tolist())
            relationships.extend(val_triples_pd["relation"].unique().tolist())
            relationships.extend(test_triples_pd["relation"].unique().tolist())
            relationships = list(set(relationships))

            # create a mapping of entities
            entity_map = {}
            logger.info(f"Creating entity mapping: {feature}")
            for index,entity in enumerate(entities):
                entity_map[entity] = index
            
            # create a mapping of relationships
            relation_map = {}
            num_entities = len(entity_map)
            logger.info(f"Creating relation mapping: {feature}")
            for index, relation in enumerate(relationships):
                relation_map[relation] = num_entities+index

            # to numpy
            train_triples_np = train_triples_pd.astype(str).to_numpy()
            train_triples_indexed_np = np.zeros(shape=train_triples_np.shape).astype(int)
            
            to_indexed_triples(train_triples_np,train_triples_indexed_np,entity_map,relation_map)

            out_path = f"{data_dir}/processed/annotated/{feature}/folds/fold_{fold_n}"

            # serialize triples
            triples_indexed_pd = pd.DataFrame(data=train_triples_indexed_np,columns=["head","relation","tail"])
            triples_indexed_pd.to_csv(f"{out_path}/train_triples_indexed.csv",index=False)

            # serialize entity mapping
            entity_map_pd = pd.DataFrame(data=list(entity_map.items()),columns=["entity","index"])
            entity_map_pd.to_csv(f"{out_path}/train_entity_map.csv",index=False)

            # serialize entity mapping
            relations_map_pd = pd.DataFrame(data=list(relation_map.items()),columns=["relation","index"])
            relations_map_pd.to_csv(f"{out_path}/train_relation_map.csv",index=False)

def to_indexed_triples(triples_named,triples_indexed,entity_map,relation_map):
    num_triples = len(triples_named)

    for index in range(num_triples):
        triple = triples_named[index]

        head_index = entity_map[triple[0]]
        rel_index = relation_map[triple[1]]
        tail_index = entity_map[triple[2]]

        triples_indexed[index][0] = head_index
        triples_indexed[index][1] = rel_index
        triples_indexed[index][2] = tail_index

if __name__ == "__main__":
    build_index()