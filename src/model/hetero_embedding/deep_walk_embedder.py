import enum
from networkx.classes.function import nodes
from ray.state import node_ids
import torch
import torch.nn as nn
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, PyTorchCallback
import torch
from torch.utils.data.dataset import TensorDataset
from src.model.utils import EarlyStopper
from src.model.word2vec import Word2VecSkipGram
import pandas as pd
from pathlib import Path
import networkx as nx
from torch_rw import utils as rw_utils
import src.misc.nv_utils as nv_utils
from src.misc.nv_utils import WalksDataSet
from torch.utils.data import random_split
from determined.pytorch import DataLoader
from loguru import logger
from determined.tensorboard.metric_writers.pytorch import TorchWriter
import src.data.go as go
from torch_rw import rw
import numpy as np
from src.callbacks.hetero_embedding.deep_walk_embedder_callback import DeepWalkCallBack
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from determined.pytorch import Reducer as reducer
import random

class DeepWalkEmbedder(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        

        self.tf_logger = TorchWriter()

        self.context = context
        self.hparams = self.context.get_hparams()
        self.data_config = self.context.get_data_config()

        # parse seed and fold
        set = self.hparams["sets"].split(",")
        self.fold = set[0]
        self.seed = int(set[1])

        # seeds
        print(f"Setting seed in trial")
        import os
        os.environ['PYTHONHASHSEED']=str(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        
        # load edges
        data = self._load_data()
        self.train_edges_tensor = data["train_edges_tensor"]
        self.node_edge_idx = data["node_edge_idx"] 
        self.nodes_tensor = data["nodes_tensor"]
        self.target_nodes_tensor = data["target_nodes_tensor"]
        self.padding_idx = data["padding_idx"]
        self.entity_id_dict = data["entity_id_dict"]
        self.protein_idx = data["protein_idx"]
        self.protein_labels = data["protein_labels"]

        # load evaluation labels
        self.labelled_data = self._load_labelled_data(self.entity_id_dict)
        self.labelled_data_ec = self._load_labelled_data_ec(self.entity_id_dict)

        # create splits
        self.train_dataset,self.val_dataset = self._create_splits(self.target_nodes_tensor)

        # init model
        self.w2vmodel = Word2VecSkipGram(num_nodes=len(self.nodes_tensor),
                                         embedding_dim=self.hparams["embedding_dim"],
                                         padding_index=self.padding_idx
                                        )

        self.model = self.context.wrap_model(self.w2vmodel)
        
        # init optimizer
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"])
        )


    def _load_data(self):
        fold_n = self.fold
        # define data dirs
        data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
        fold_path = f"{data_dir}/processed/folds/fold_{fold_n}"
        fold_path_antd = f"{data_dir}/processed/annotated/folds/fold_{fold_n}"

        logger.info(f"Reading triples from: {fold_path}")

        # read the triples
        train_triples_indexed_pd = pd.read_csv(f"{fold_path_antd}/train_triples_indexed.csv")
        train_triples_indexed_pd = train_triples_indexed_pd.drop(columns=["relation"])
        train_edges_tensor = torch.from_numpy(train_triples_indexed_pd.to_numpy())
                
        # read the entity mapping
        train_entity_mapping_pd = pd.read_csv(f"{fold_path_antd}/train_entity_map.csv")
        train_entity_mapping_pd = train_entity_mapping_pd.sort_values(by="index",ascending=True)

        # convert to dict with entity:index pairs
        logger.info(f"Creating node index mapping dict")
        entity_id_dict = train_entity_mapping_pd.set_index("entity").to_dict()['index']

        # nodes tensor
        nodes_tensor = torch.Tensor(train_entity_mapping_pd["index"]).to(int)

        # build node edge index
        logger.info("Creating node edge index mapping")
        node_edge_idx, train_edges_tensor_sorted = rw_utils.build_node_edge_index(train_edges_tensor,nodes_tensor)

        # create the padding index
        padding_index = nodes_tensor[-1] + 1
        logger.info(f"Padding idx is : {padding_index}")

        # create target nodes tensor
        logger.info("Creating target nodes tensor")
        proteins = pd.read_csv(f"{data_dir}/processed/proteins.csv")["protein"].tolist()
        protein_idxs = []
        for protein in proteins:
            idx = entity_id_dict[protein]
            protein_idxs.append(idx)

        target_nodes_tensor = nodes_tensor
        protein_idx_tensor = torch.Tensor(protein_idxs).to(int)
        
        # move to gpu
        train_edges_tensor_sorted = train_edges_tensor_sorted.cuda()
        node_edge_idx = node_edge_idx.cuda()
        nodes_tensor = nodes_tensor.cuda()
        target_nodes_tensor = target_nodes_tensor.cuda()
        protein_idx_tensor = protein_idx_tensor.cuda()

        data = {
            "train_edges_tensor": train_edges_tensor_sorted,
            "node_edge_idx": node_edge_idx,
            "nodes_tensor": nodes_tensor,
            "target_nodes_tensor": target_nodes_tensor,
            "padding_idx": padding_index,
            "entity_id_dict": entity_id_dict,
            "protein_idx": protein_idx_tensor,
            "protein_labels": proteins
        }

        return data

    def _load_labelled_data(self,entity_id_dict):

        fold_n = self.fold

        data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
        
        # train edges
        pos_edges_train = pd.read_csv(f"{data_dir}/processed/edges/fold_{fold_n}/positive_edges_train.csv").drop(columns=["label"])
        neg_edges_train = pd.read_csv(f"{data_dir}/processed/edges/fold_{fold_n}/negative_edges_train.csv").drop(columns=["label"])

        # val edges
        pos_edges_val = pd.read_csv(f"{data_dir}/processed/edges/fold_{fold_n}/positive_edges_val.csv").drop(columns=["label"])
        neg_edges_val = pd.read_csv(f"{data_dir}/processed/edges/fold_{fold_n}/negative_edges_val.csv").drop(columns=["label"])

        # map entities to their ids
        logger.info("Mapping edges to entity ids")
        pos_edges_train = self._to_indexed(pos_edges_train,entity_id_dict)
        pos_edges_val = self._to_indexed(pos_edges_val,entity_id_dict)
        neg_edges_train = self._to_indexed(neg_edges_train,entity_id_dict)
        neg_edges_val = self._to_indexed(neg_edges_val,entity_id_dict)

        # to numpy
        logger.info("To numpy")
        pos_edges_train = pos_edges_train.to_numpy()
        pos_edges_val = pos_edges_val.to_numpy()
        neg_edges_train = neg_edges_train.to_numpy()
        neg_edges_val = neg_edges_val.to_numpy()

        # labels
        logger.info("creating labels")
        pos_labels_train = np.ones((len(pos_edges_train)))
        pos_labels_val = np.ones((len(pos_edges_val)))
        neg_labels_train = np.zeros((len(neg_edges_train)))
        neg_labels_val = np.zeros((len(neg_edges_val)))    

        # combine edges
        logger.info("Combining edges")
        train_edges = np.concatenate((pos_edges_train,neg_edges_train),axis=0)
        val_edges = np.concatenate((pos_edges_val,neg_edges_val),axis=0)
        
        # combine labels
        logger.info("Combining labels")
        train_labels = np.concatenate((pos_labels_train,neg_labels_train))
        val_labels = np.concatenate((pos_labels_val,neg_labels_val))

        # shuffle
        logger.info("Shufffling")
        from sklearn.utils import shuffle
        train_edges, train_labels = shuffle(train_edges,train_labels,random_state=self.seed)
        val_edges, val_labels = shuffle(val_edges,val_labels,random_state=self.seed)

        # subset
        train_edges = train_edges[:6000]
        train_labels = train_labels[:6000]
        val_edges = val_edges[:1000]
        val_labels = val_labels[:1000]

        # edges to torch
        train_edges = torch.from_numpy(train_edges).cuda()
        val_edges = torch.from_numpy(val_edges).cuda()

        # labels to torch
        train_labels = torch.from_numpy(train_labels)
        val_labels = torch.from_numpy(val_labels)

        data = {
            "train_edges": train_edges,
            "val_edges": val_edges,
            "train_labels": train_labels,
            "val_labels": val_labels
        }

        return data

    def _load_labelled_data_ec(self,entity_id_dict):

        data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"
        labelled_path = f"{data_dir}/processed/ec_labels.csv"

        logger.info(f"Reading labelled data from: {labelled_path}")

        labelled_data_df = pd.read_csv(labelled_path)

        # node indices
        node_indices = []
        evaluation_nodes = labelled_data_df["protein"].to_list()
        for eval_node in evaluation_nodes:
            idx = entity_id_dict[eval_node]
            node_indices.append(idx)

        node_indices = torch.Tensor(node_indices).long().cuda()

        # get labels
        labels = labelled_data_df["ec_label"].to_numpy()
        labels = torch.from_numpy(labels).int()

        data = {
            "node_indices": node_indices,
            "labels": labels
        }

        return data


    def _to_indexed(self,edges_df,entity_map):
        edges_indexed = []
        for _, row in tqdm(edges_df.iterrows(),total=len(edges_df)):
            head = row["head"]
            tail = row["tail"]

            head_index = entity_map[head]
            tail_index = entity_map[tail]
            edges_indexed.append((head_index,tail_index))

        edges_indexed_df = pd.DataFrame(data=edges_indexed,columns=["head","tail"])
        return edges_indexed_df

    def build_callbacks(self):
        fold_n = self.fold
        return {"callback": DeepWalkCallBack(trial = self,
                                        early_stopping_delta=self.hparams["early_stopping_delta"],
                                        early_stopping_patience=self.hparams["early_stopping_patience"],
                                        protein_idx=self.protein_idx,
                                        fold_n=fold_n,
                                        protein_labels = self.protein_labels
                                    )}

    def train_batch(self, batch, epoch_idx: int, batch_idx: int):
        #logger.info(f"Epoch: {epoch_idx} / Batch idx: {batch_idx} / Batch len: {len(batch[0])}")
        target_nodes = batch[0]
        pos_context = batch[1]
        neg_context = batch[2]

        # get the loss
        loss = self.model(target_nodes,pos_context,neg_context)

        # backward pass
        self.context.backward(loss)
        
        # step the optimizer
        self.context.step_optimizer(self.optimizer,
                                    clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(params, 0.5, error_if_nonfinite=True))

        return {"train_loss":loss}

    def evaluate_batch(self, batch,batch_idx):
        target_nodes = batch[0]
        pos_context = batch[1]
        neg_context = batch[2]

        # calc val loss
        loss = self.model(target_nodes,pos_context,neg_context)

        roc_score_link_pred = 0
        roc_score_clf = 0

        if batch_idx == 0:
            roc_score_link_pred = self.evaluate_au_roc_link_pred()
            roc_score_clf = self.evaluate_au_roc_clf()
        else:
            roc_score_link_pred = 0
            roc_score_clf = 0

        metrics = {
            "val_loss": loss,
            "roc_score_link_pred": roc_score_link_pred,
            "roc_score_clf": roc_score_clf  
        }
        
        return metrics

    def evaluation_reducer(self):
        reducers = {
            "val_loss": reducer.AVG,
            "roc_score_link_pred": reducer.MAX,
            "roc_score_clf": reducer.MAX 
        }

        return reducers

    def evaluate_au_roc_link_pred(self):

        # data
        train_edges = self.labelled_data["train_edges"]
        val_edges = self.labelled_data["val_edges"]
        train_labels = self.labelled_data["train_labels"].numpy()
        val_labels = self.labelled_data["val_labels"].numpy()

        # get embeddings
        embeddings = self.model.get_embedding(to_cpu=False)

        # get edge embeddings train
        logger.info("Creating edge embeddings train")
        head_embedding_train = embeddings[train_edges[:,0]]
        tail_embedding_train = embeddings[train_edges[:,1]]
        edge_embedding_train = head_embedding_train * tail_embedding_train

        # get edge embeddings val
        logger.info("Creating edge embeddings val")
        head_embedding_val = embeddings[val_edges[:,0]]
        tail_embedding_val = embeddings[val_edges[:,1]]
        edge_embedding_val = head_embedding_val * tail_embedding_val

        # move embedding to cpu and convert to numpy
        edge_embedding_train = edge_embedding_train.cpu().numpy()
        edge_embedding_val = edge_embedding_val.cpu().numpy()

        logger.info("Training edge predictor")
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=self.seed).fit(edge_embedding_train,train_labels)

        logger.info("Predicting")
        y_predicted_proba = clf.predict_proba(edge_embedding_val)

        # calculate avg precision
        au_roc = roc_auc_score(val_labels,y_predicted_proba[:,1])

        return au_roc

    def evaluate_au_roc_clf(self):

        node_indices = self.labelled_data_ec["node_indices"]
        labels = self.labelled_data_ec["labels"].numpy()

        # get embeddings
        embeddings = self.model.get_embedding(to_cpu=False)
        embedding_protein_nodes = embeddings[node_indices].cpu().numpy()
        
        # split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(embedding_protein_nodes,
                                                            labels,test_size=0.25,random_state=self.seed,shuffle=True)

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=self.seed).fit(X_train,y_train)

        y_predicted_proba = clf.predict_proba(X_test)
        
        # calculate avg precision
        au_roc = roc_auc_score(y_test,y_predicted_proba[:,1])

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

    def sample(self,nodes_batch):

        batch_len = len(nodes_batch)

        nodes_batch = torch.Tensor(nodes_batch).squeeze(dim=1).long().contiguous().to(self.context.device)

        assert len(nodes_batch) == batch_len

        nodes_batch = nodes_batch.repeat_interleave(self.hparams["walks_per_node"],0)

        idx = torch.randperm(nodes_batch.nelement())
        nodes_batch = nodes_batch.view(-1)[idx].view(nodes_batch.size()).contiguous()

        # generate walks
        walks = rw.walk_edge_list(
            self.train_edges_tensor,
            self.node_edge_idx,
            target_nodes=nodes_batch,
            p = self.hparams["p"],
            q = self.hparams["q"],
            padding_idx=self.padding_idx,
            walk_length=self.hparams["walk_length"],
            restart=self.hparams["restart"],
            seed=self.seed
        )

        # repeat walks
        walks = walks.repeat_interleave(self.hparams["num_negatives"],0)


        # generate contexts
        target_nodes,pos_context,neg_context = rw.to_windows(walks=walks,
                                                        window_size=self.hparams["context_size"],
                                                        num_nodes=len(self.nodes_tensor),
                                                        seed=self.seed
                                                    )
    
        return target_nodes,pos_context,neg_context

    
    
    def _create_splits(self,nodes):
        nodes_dataset = TensorDataset(nodes)
        train_length = int(len(nodes_dataset) * 0.8)
        val_length = len(nodes_dataset) - train_length
        logger.info(f"Train len: {train_length}")
        train_dataset,val_dataset = random_split(dataset=nodes_dataset,
                                                lengths=[train_length,val_length])

        return train_dataset,val_dataset


    


    





