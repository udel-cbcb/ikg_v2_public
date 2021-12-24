import enum
from networkx.classes.function import nodes
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
from src.callbacks.protein_callback import ProteinCallBack
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class ProteinEmbedder(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        
        torch.manual_seed(20)
        np.random.seed(20) 

        self.tf_logger = TorchWriter()

        self.context = context
        self.hparams = self.context.get_hparams()
        
        # load graph data
        self.row_ptr,self.col_idx,self.nodes_tensor,self.protein_graph = self._load_data()

        # load labelled data
        self.labelled_data_df = self._load_labelled_data()

        # create splits
        self.train_dataset,self.val_dataset = self._create_splits(self.nodes_tensor)

        # init model
        self.w2vmodel = Word2VecSkipGram(num_nodes=len(self.protein_graph.nodes()),
                                         embedding_dim=self.hparams["embedding_dim"])

        self.model = self.context.wrap_model(self.w2vmodel)
        
        # init optimizer
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"])
        )

    def build_callbacks(self):
        return {"callback": ProteinCallBack(trial = self,
                                        node_labels=list(self.protein_graph.nodes()),
                                        fold_n=self.hparams["fold_n"],
                                        early_stopping_delta=self.hparams["early_stopping_delta"],
                                        early_stopping_patience=self.hparams["early_stopping_patience"]
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

    def evaluate_batch(self, batch):
        target_nodes = batch[0]
        pos_context = batch[1]
        neg_context = batch[2]

        # calc val loss
        loss = self.model(target_nodes,pos_context,neg_context)
        au_roc = self.evaluate_au_roc()
                
        metrics = {
            "val_loss": loss,
            "clf_au_roc": au_roc
        }
        

        return metrics

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

    def _load_data(self):
        fold_n = self.hparams["fold_n"]

        data_dir = f"{Path.home()}/data/ikg_v2_data/hetero_embedding"
        fold_path = f"{data_dir}/processed/folds/fold_{fold_n}"
        fold_path_ann = f"{data_dir}/processed/annotated/folds/fold_{fold_n}"


        logger.info(f"Reading protein graph from: {fold_path}")

        # read the triples
        train_triples_pd = pd.read_csv(f"{fold_path_ann}/train_edges.csv")
        val_triples_pd = pd.read_csv(f"{fold_path}/val_triples.csv")
        test_triples_pd = pd.read_csv(f"{fold_path}/test_triples.csv")

        # create graph from train edges
        protein_graph = nx.Graph()
        self._add_edges_to_graph(graph=protein_graph,triples_pd=train_triples_pd)

        # add nodes from val and test edges
        self._add_nodes_to_graph(graph=protein_graph,triples_pd=val_triples_pd)
        self._add_nodes_to_graph(graph=protein_graph,triples_pd=test_triples_pd)

        # graph stats
        logger.info(f"Number of nodes:{len(protein_graph.nodes())}")
        logger.info(f"Number of edges:{len(protein_graph.edges())}")

        # convert the graph to undirected
        protein_graph = nx.to_undirected(protein_graph)          

        # get csr representation
        row_ptr, col_idx = rw_utils.to_csr(protein_graph)
        nodes_tensor = rw_utils.nodes_tensor(protein_graph)
        
        # move to gpu
        row_ptr = row_ptr.to(self.context.device)
        col_idx = col_idx.to(self.context.device)
        nodes_tensor = nodes_tensor.to(self.context.device)

        return row_ptr,col_idx,nodes_tensor,protein_graph

    
    def _create_splits(self,nodes):
        nodes_dataset = TensorDataset(nodes)
        train_length = int(len(nodes_dataset) * 0.8)
        val_length = len(nodes_dataset) - train_length
        logger.info(f"Train len: {train_length}")
        train_dataset,val_dataset = random_split(dataset=nodes_dataset,
                                                lengths=[train_length,val_length])

        return train_dataset,val_dataset


    def sample(self,nodes_batch):

        batch_len = len(nodes_batch)

        nodes_batch = torch.Tensor(nodes_batch).squeeze(dim=1).long().contiguous().to(self.context.device)

        assert len(nodes_batch) == batch_len

        nodes_batch = nodes_batch.repeat_interleave(self.hparams["walks_per_node"],0)
        
        # generate walks
        walks = rw.walk(
            self.row_ptr,
            self.col_idx,
            target_nodes=nodes_batch,
            p = self.hparams["p"],
            q = self.hparams["q"],
            walk_length=self.hparams["walk_length"],
            seed=20
        )

        # repeat walks
        walks = walks.repeat_interleave(self.hparams["num_negatives"],0)

        # generate contexts
        target_nodes,pos_context,neg_context = rw.to_windows(walks=walks,
                                                        window_size=self.hparams["context_size"],
                                                        num_nodes=len(self.nodes_tensor),
                                                        seed=20
                                                    )
    
        return target_nodes,pos_context,neg_context

    def _load_labelled_data(self):

        data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"
        labelled_path = f"{data_dir}/processed/ec_labels.csv"

        logger.info(f"Reading labelled data from: {labelled_path}")

        labelled_data_df = pd.read_csv(labelled_path)

        return labelled_data_df

    def evaluate_au_roc(self):

        # get node indices
        nodes = list(self.protein_graph.nodes())
        node_indices = []
        evaluation_nodes = self.labelled_data_df["protein"].to_list()
        for eval_node in evaluation_nodes:
            idx = nodes.index(eval_node)
            node_indices.append(idx)

        # get embeddings
        embeddings = self.model.get_embedding().numpy()
        embedding_protein_nodes = embeddings[node_indices]
        
        # get labels
        labels = self.labelled_data_df["ec_label"].to_numpy()

        # split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(embedding_protein_nodes,
                                                            labels,test_size=0.25,random_state=20,shuffle=True)

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=20).fit(X_train,y_train)

        y_predicted_proba = clf.predict_proba(X_test)
        
        # calculate avg precision
        au_roc = roc_auc_score(y_test,y_predicted_proba[:,1])

        return au_roc


    def _add_edges_to_graph(self,graph,triples_pd):
        for index, row in tqdm(triples_pd.iterrows(),total=len(triples_pd)):
            graph.add_edge(row["head"],row["tail"])


    def _add_nodes_to_graph(self,graph,triples_pd):
        graph.add_nodes_from(triples_pd["head"])
        graph.add_nodes_from(triples_pd["tail"])





