from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
import torch
import numpy as np
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from src.model.conv_kb import ConvKB
from loguru import logger
from pathlib import Path
import pandas as pd
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score
from src.callbacks.hetero_embedding.conv_kb_callback import KBCallback

class ConvKBEmbedder(PyTorchTrial):

    def __init__(self, context: PyTorchTrialContext):
        
        torch.manual_seed(20)
        np.random.seed(20) 

        self.tf_logger = TorchWriter()

        # context
        self.context = context
        self.hparams = self.context.get_hparams()
        self.data_config = self.context.get_data_config()

        # load data
        data = self._load_data()
        self.train_triples = data["train_triples"]
        self.val_triples = data["val_triples"]
        self.num_entities = data["num_entities"]
        self.num_relations = data["num_relations"]
        self.entity_id_mapping = data["entity_mapping"]
        self.relation_mapping = data["relation_mapping"]

        # load labelled data
        self.labelled_data = self._load_labelled_data(entity_id_mapping=self.entity_id_mapping,
                                                        relation_mapping=self.relation_mapping)
        
        # create splits
        self.train_dataset,self.val_dataset = self._create_datasets(self.train_triples)

        # init model
        self.conv_kb_model = ConvKB(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            embedding_dim=self.hparams["embedding_dim"],
            out_channels=self.hparams["out_channels"],
            kernel_size=self.hparams["kernel_size"],
            dropout=self.hparams["dropout"],
            lmbda=self.hparams["lmbda"]
        )

        self.model = self.context.wrap_model(self.conv_kb_model)
        
        # init optimizer
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"])
        )

    def _load_data(self):
        fold_n = self.data_config["fold_n"]

        # define data dirs
        data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
        fold_path = f"{data_dir}/processed/folds/fold_{fold_n}"
        fold_path_antd = f"{data_dir}/processed/annotated/folds/fold_{fold_n}"
        fold_path_edges = f"{data_dir}/processed/edges/fold_{fold_n}"

        logger.info(f"Reading triples from: {fold_path}")

        # read the train triples
        train_triples_indexed_pd = pd.read_csv(f"{fold_path_antd}/train_triples_indexed.csv")

        # read the entity mapping
        entity_mapping_pd = pd.read_csv(f"{fold_path_antd}/train_entity_map.csv")
        entity_mapping_pd = entity_mapping_pd.sort_values(by="index",ascending=True)

        # read the entity mapping
        relation_mapping_pd = pd.read_csv(f"{fold_path_antd}/train_relation_map.csv")
        relation_mapping_pd = relation_mapping_pd.sort_values(by="index",ascending=True)

        # remove inverse relations
        has_a_id = relation_mapping_pd[relation_mapping_pd["relation"] == "has_a"].iloc[0]["index"]
        annotated_as_bp_to_id = relation_mapping_pd[relation_mapping_pd["relation"] == "annotated_as_bp_to"].iloc[0]["index"]
        annotated_as_bp_to_id = relation_mapping_pd[relation_mapping_pd["relation"] == "annotated_as_mf_to"].iloc[0]["index"]
        p_id = relation_mapping_pd[relation_mapping_pd["relation"] == "p"].iloc[0]["index"]

        """train_triples_indexed_pd = train_triples_indexed_pd[train_triples_indexed_pd["relation"] != has_a_id]
        train_triples_indexed_pd = train_triples_indexed_pd[train_triples_indexed_pd["relation"] != annotated_as_bp_to_id]
        train_triples_indexed_pd = train_triples_indexed_pd[train_triples_indexed_pd["relation"] != annotated_as_bp_to_id]"""
        
        #train_triples_indexed_pd = train_triples_indexed_pd[train_triples_indexed_pd["relation"] == p_id]


        #to tensor
        train_triples_tensor = torch.Tensor(train_triples_indexed_pd.to_numpy())

        # shuffle
        idx = torch.randperm(len(train_triples_tensor))
        train_triples_tensor = train_triples_tensor[idx][:5000]

        # replace relations ids to start from 0
        relation_id_0 = relation_mapping_pd.iloc[0]["index"]
        train_triples_tensor[:,1] = train_triples_tensor[:,1] - relation_id_0
        relation_mapping_pd["index"] = relation_mapping_pd["index"].apply(lambda x: x - relation_id_0)
             
        # convert to dict with entity:index pairs
        logger.info(f"Creating node index mapping dict")
        entity_id_dict = entity_mapping_pd.set_index("entity").to_dict()['index']

        # convert to dict with relation:index pairs
        logger.info(f"Creating node index mapping dict")
        relation_id_dict = relation_mapping_pd.set_index("relation").to_dict()['index']
                               
        # load val edges
        val_edges_pd = pd.read_csv(f"{fold_path_edges}/all_edges_val.csv")

        # convert val_edges to indexed val triples
        val_triples_tensor = self._to_indexed_val_triples(val_edges_pd,entity_id_dict,relation_id_dict)
        

        data = {
            "train_triples": train_triples_tensor,
            "val_triples": val_triples_tensor,
            "entity_mapping": entity_id_dict,
            "relation_mapping": relation_id_dict,
            "num_entities": len(entity_id_dict),
            "num_relations": len(relation_id_dict)
        }

        return data

    def _load_labelled_data(self,entity_id_mapping,relation_mapping):

        fold_n = self.data_config["fold_n"]

        data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
        
        # val edges
        all_edges_val = pd.read_csv(f"{data_dir}/processed/edges/fold_{fold_n}/all_edges_val.csv").drop(columns=["source"])
                        
        # to edges and labels
        logger.info("To torch")
        edges_val = all_edges_val[["head","tail"]]
        labels_val = torch.from_numpy(all_edges_val[["label"]].to_numpy())

        # to triples
        triples_val = self._to_indexed_val_triples(edges_val,entity_id_mapping,relation_mapping).cuda()

        # shuffle
        logger.info("Shufffling")
        from sklearn.utils import shuffle
        triples_val, labels_val = shuffle(triples_val,labels_val,random_state=20)
        

        data = {
            "triples": triples_val,
            "labels": labels_val,
        }

        return data

    def _to_indexed_val_triples(self,val_edges,entity_id_mapping,relation_id_mapping):
        num_val_edges = len(val_edges)
        val_edges_indexed = torch.empty(size=(num_val_edges,3))
        for index, val_edge in val_edges.iterrows():
            head = val_edge["head"]
            relation = "p"
            tail = val_edge["tail"]

            head_index = entity_id_mapping[head]
            relation_index = relation_id_mapping[relation]
            tail_index = entity_id_mapping[tail]
            
            val_edges_indexed[index][0] = head_index 
            val_edges_indexed[index][1] = relation_index 
            val_edges_indexed[index][2] = tail_index

        val_edges_indexed = val_edges_indexed.to(int) 

        return val_edges_indexed

    def train_batch(self, batch, epoch_idx: int, batch_idx: int):
        #logger.info(f"Epoch: {epoch_idx} / Batch idx: {batch_idx} / Batch len: {len(batch[0])}")
        triples = batch[0]
        labels = batch[1]

        # get the loss
        loss = self.model(triples,labels)

        # backward pass
        self.context.backward(loss)
        
        # step the optimizer
        self.context.step_optimizer(self.optimizer)

        return {"train_loss":loss}

    def evaluate_batch(self, batch,batch_idx):
        triples = batch[0]
        labels = batch[1]

        # calc val loss
        loss = self.model(triples,labels)

        roc_link_pred = self.evaluate_au_roc_link_pred()

        metrics = {
            "val_loss": loss,
            "roc_link_pred": roc_link_pred
        }
        
        return metrics

    def evaluate_au_roc_link_pred(self):

        # data
        val_triples = self.labelled_data["triples"]
        val_labels = self.labelled_data["labels"].cpu().numpy()

        # get embeddings
        probs = self.model.predict(val_triples).cpu().numpy()

        # calculate avg precision
        au_roc = roc_auc_score(val_labels,probs)

        return au_roc


    def build_training_data_loader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.context.get_per_slot_batch_size(),
                          shuffle=True,
                          collate_fn=self.sample,
                          drop_last=True,
                        )
        

    def build_validation_data_loader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.context.get_per_slot_batch_size(),
                          collate_fn=self.sample
                        )

    def sample(self,triples_batch_positive):
        
        triples_batch_positive = torch.Tensor(triples_batch_positive).to(int).cuda()

        triples_batch_positive = triples_batch_positive.repeat_interleave(self.hparams["num_negatives"],0)

        # corrupt the tail and create negatives
        triples_batch_negative = torch.clone(triples_batch_positive).cuda()

        # corrupt half heads and half tails
        head_idx = torch.randperm(int(len(triples_batch_negative)/2))
        triples_batch_negative[head_idx,0] = torch.randint(low=0,high=self.num_entities-1,size=(len(triples_batch_negative[head_idx]),)).cuda()
        triples_batch_negative[~head_idx,2] = torch.randint(low=0,high=self.num_entities-1,size=(len(triples_batch_negative[~head_idx]),)).cuda()
        
        # create labels
        positive_labels = torch.ones(size=(len(triples_batch_positive),1)).cuda()
        negative_labels = torch.empty(size=(len(triples_batch_negative),1)).fill_(-1).cuda()

        # concat
        triples_batch = torch.cat((triples_batch_positive,triples_batch_negative),dim=0)
        labels_batch = torch.cat((positive_labels,negative_labels),dim=0)

        # shuffle
        idx = torch.randperm(len(triples_batch))
        triples_batch = triples_batch[idx].contiguous()
        labels_batch = labels_batch[idx].contiguous()

        return triples_batch,labels_batch

    def _create_datasets(self,triples):
        triples_dataset = TensorDataset(triples[:,0],triples[:,1],triples[:,2])

        train_length = int(len(triples_dataset) * 0.75)
        val_length = len(triples_dataset) - train_length
        logger.info(f"Train len: {train_length}")
        train_dataset,val_dataset = random_split(dataset=triples_dataset,
                                                lengths=[train_length,val_length])

        return train_dataset,val_dataset

    def build_callbacks(self):
        return {"callback": KBCallback(trial = self,
                                        early_stopping_delta=self.hparams["early_stopping_delta"],
                                        early_stopping_patience=self.hparams["early_stopping_patience"],
                                    )}
    
    