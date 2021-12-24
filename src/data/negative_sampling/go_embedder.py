import obonet
import torch
from loguru import logger
from tqdm.std import tqdm
from src.data.iptmnet import build_graph
from src.model.node2vec import Node2VecDataLoader, Word2VecSkipGram, WalksDataSet, ContextDataSet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
import src.model.utils as mu
import pandas as pd
from tqdm import tqdm
import networkx as nx
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import random_split
from pathlib import Path

def _train_emb(
    go_graph,
    embedding_dim,
    walk_length,
    walks_per_node,
    context_size,
    num_negative_samples,
    rw_batch_size,
    walk_batch_size,
    wv_batch_size,
    num_epochs,
    patience,
    early_stopping_delta,
    learning_rate,
    stop_at_batch,
    term_type,
    seed,
    device
):
    torch.manual_seed(seed)
    
    # graph stats
    num_nodes = len(go_graph.nodes())
    num_edges = len(go_graph.edges())

    logger.info(f"Number of nodes:{num_nodes}")
    logger.info(f"Number of edges:{num_edges}")

    # perform random walks on the graph
    logger.info("Generating walks")
    walks_tensor = nv.generate_walks(graph=go_graph,
                                     p=1.0,
                                     q=1.0,
                                     walk_length=walk_length-1,
                                     walks_per_node=walks_per_node,
                                     batch_size=rw_batch_size,
                                     device=device,
                                     seed=seed   
                                    )

    # create dataset
    walks_dataset = WalksDataSet(walks_tensor)
   
    # create the walk data loader
    walk_dataloader = DataLoader(dataset=walks_dataset,batch_size=walk_batch_size,shuffle=True,num_workers=8,prefetch_factor=2)

    # create the model
    logger.info("Creating model")
    model = Word2VecSkipGram(
        embedding_dim=embedding_dim,
        num_nodes=len(go_graph.nodes()),
        learning_rate=learning_rate
    )
    model = model.to(device)

    # get the trainer
    tb_logger = TensorBoardLogger("runs", name=f"go_embedder_{term_type}")
    early_stopping = EarlyStopping(monitor="val_loss",
                                   min_delta=early_stopping_delta,
                                   patience=patience,
                                   verbose=True,
                                   mode="min")
    trainer = pl.Trainer(logger=tb_logger,callbacks=early_stopping,gpus=1,max_epochs=1)

    logger.info("Started training")

    is_stopped_at_batch=False
    is_early_stopped = False
    batch_index = 0
    for _ in tqdm(range(num_epochs),total=num_epochs,disable=False):

        for walks in tqdm(walk_dataloader,total=len(walk_dataloader), disable=False):
            
            walks = walks.to(device)
            walks = walks.repeat_interleave(num_negative_samples,0)

            # to context
            target_nodes, pos_context, neg_context = nv.to_context(walks=walks,
                                                                context_size=context_size,
                                                                num_nodes=num_nodes)

            # create context dataset
            context_train_dataset, context_val_dataset = random_split(dataset=ContextDataSet(target_nodes,pos_context,neg_context),
                                           lengths=[int(len(target_nodes)*0.8),int(len(target_nodes)*0.2)])

            context_train_dataloader = DataLoader(dataset=context_train_dataset,batch_size=wv_batch_size,shuffle=False)
            context_val_dataloader = DataLoader(dataset=context_val_dataset,batch_size=wv_batch_size,shuffle=False)
            
            # perform training
            logger.info(early_stopping.stopped_epoch)
            trainer.fit(model, train_dataloaders=context_train_dataloader,val_dataloaders=context_val_dataloader)
            
            # stop at this batch if required
            if stop_at_batch != -1 and stop_at_batch == batch_index:
                is_stopped_at_batch = True
                break

            # stop if trainer was interrupted
            if early_stopping.false == True:
                is_early_stopped = True
                break
            
            print(batch_index)

            # update batch index            
            batch_index = batch_index + 1

        print(batch_index)
        if is_stopped_at_batch == True:
            logger.warning(f"Stopping at batch : {batch_index}")
            break

        if is_early_stopped == True:
            logger.warning(f"Early stopped at : {batch_index}")
            break

    # create embedding
    logger.info("Creating embeddings dataframe")
    embedding = model.get_embedding().numpy()
    embedding_df = pd.DataFrame(data=embedding)
    embedding_df.insert(0,"node",go_graph.nodes())

    return embedding_df
    

def build(data_dir,term_type):
    # load go ontology
    go_graph = nx.read_gml(f"{data_dir}/processed/go_{term_type}_graph.gml")
    go_graph = go_graph.to_undirected()

    go_embedding = _train_emb(
        go_graph = go_graph,
        embedding_dim=256,
        walk_length=500,
        walks_per_node=20,
        context_size=10,
        num_negative_samples=5,
        rw_batch_size=4000,
        walk_batch_size=128,
        wv_batch_size=512,
        num_epochs=5,
        patience=20,
        early_stopping_delta=0.001,
        learning_rate=0.0001,
        stop_at_batch=-1,
        seed=20,
        term_type=term_type,
        device="cuda"
    )
    go_embedding.to_csv(f"{data_dir}/processed/go_embedding_{term_type}.csv",index=False)

def test():
    data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"
    go_graph = nx.read_gml(f"{data_dir}/processed/go_cc_graph.gml")
    go_graph = go_graph.to_undirected()

    go_embedding = _train_emb(
        go_graph = go_graph,
        embedding_dim=256,
        walk_length=500,
        walks_per_node=20,
        context_size=10,
        num_negative_samples=5,
        rw_batch_size=4000,
        walk_batch_size=128,
        wv_batch_size=512,
        num_epochs=5,
        patience=5,
        early_stopping_delta=0.1,
        learning_rate=0.0001,
        stop_at_batch=-1,
        seed=20,
        term_type="cc",
        device="cuda"
    )

if __name__ == "__main__":
    test()
