import networkx as nx
import pandas as pd
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core import datamodule
import model.node2vec as nv
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from model.node2vec import Node2VecDataLoader, Word2VecSkipGram, WalksDataSet
import torch
from tqdm import tqdm
from misc import plotter
from model.data_loader import FastTensorDataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from model.utils import EarlyStopper
import copy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from torch.utils.data import random_split
from pathlib import Path
from pytorch_lightning.plugins import DDPPlugin
from model.data_module import Node2VecDataModule

def train(
    embedding_dim,
    walk_length,
    walks_per_node,
    p,
    q,
    context_size,
    num_negative_samples,
    rw_batch_size,
    num_epochs,
    patience,
    early_stopping_delta,
    learning_rate
):

    torch.manual_seed(20)

    data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"

    # load the data
    edges_raw = pd.read_csv(f"{data_dir}/raw/test/cora/cora.cites",sep='\t',header=None)
    print(edges_raw.head())

    # load the labels
    node_labels_raw = pd.read_csv(f"{data_dir}/raw/test/cora/cora.content",sep='\t',header=None)
    node_labels = pd.DataFrame()
    node_labels["node"] = node_labels_raw.to_numpy()[:,0]
    node_labels["label"] = node_labels_raw.to_numpy()[:,-1]
    node_labels = node_labels.sort_values(by=["node"])

    # create graph
    email_graph = nx.from_edgelist(edges_raw.to_numpy())
    print(node_labels.head())
    email_nodes = list(email_graph.nodes())
    print(email_nodes[:10])

    # sort nodes according to order in graph
    node_idx = []
    for _, item in node_labels.iterrows():
        node = item["node"]
        idx = email_nodes.index(node)
        node_idx.append(idx)

    node_labels["node_idx"] = node_idx
    node_labels = node_labels.sort_values(by=["node_idx"],ascending=True)   


    # create data module
    dm = Node2VecDataModule(graph=email_graph,
                            batch_size=32,
                            rw_batch_size=rw_batch_size,
                            walk_length=walk_length,
                            p=p,
                            q=q,
                            num_negatives=num_negative_samples,
                            context_size=context_size,
                            walks_per_node=walks_per_node
    )

    # create the model
    model = Word2VecSkipGram(
        embedding_dim=embedding_dim,
        num_nodes=len(email_graph.nodes()),
        learning_rate=learning_rate
    )
    
    # get the trainer
    tb_logger = TensorBoardLogger("runs", name=f"n2vec_embedder_test")
    early_stopping = EarlyStopping(monitor="val_loss",
                                   min_delta=early_stopping_delta,
                                   patience=patience,
                                   verbose=True,
                                   mode="min")
    
    trainer = pl.Trainer(logger=tb_logger,
                        callbacks=early_stopping,
                        gpus=-1,
                        val_check_interval=500)

    logger.info("Starting training")
    
    trainer.fit(model,dm)
        
    # get embeddings
    nodes = torch.Tensor(list(email_graph.nodes())).int()
    embeddings = model.get_embedding().numpy()

    # plot
    logger.info("Plotting embeddings")
    plotter.plot_and_save_embeddings(embeddings,nodes,node_labels,"results/figures/email_nodes.png")



from knockknock import slack_sender
import data.constants as constants
@slack_sender(webhook_url=constants.webhook_url, channel="deep_learning")
def test():
    train(
        embedding_dim=128,
        walk_length=100,
        walks_per_node=10,
        p=1.0,
        q=1.0,
        context_size=10,
        num_negative_samples=10,
        rw_batch_size=8096,
        num_epochs=10,
        patience=10,
        early_stopping_delta=0.002,
        learning_rate=0.001
    )



