import networkx as nx
from loguru import logger
from torch_geometric.nn import Node2Vec
import torch
import obonet
from tqdm import tqdm
import pandas as pd
from misc import plotter
from collections import deque
from model.utils import EarlyStopper
import copy

def build(seed=10,
          embedding_dim=128,
          walk_length=100,
          context_size=10,
          walks_per_node=10,
          num_negative_samples=10,
          p=1.0,
          q=0.7,
          sparse=True,
          save=True,
          num_epochs=500,
          patience=10,
          early_stopping_delta=0.002,
          show_progress=True,
          device=None,
          subsample=False,
        ):

    torch.manual_seed(seed)

    from pathlib import Path

    data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"

    # load the data
    edges_raw = pd.read_csv(f"{data_dir}/raw/test/cora/cora.cites",sep='\t',header=None)

    # load the labels
    node_labels_raw = pd.read_csv(f"{data_dir}/raw/test/cora/cora.content",sep='\t',header=None)
    node_labels = pd.DataFrame()
    node_labels["node"] = node_labels_raw.to_numpy()[:,0]
    node_labels["label"] = node_labels_raw.to_numpy()[:,-1]
    node_labels = node_labels.sort_values(by=["node"])
    print(node_labels.head())

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

    # convert to pytorch-geomtric dataset
    from torch_geometric.utils import from_networkx
    logger.info("Converting to torch_geomtric data")
    data = from_networkx(email_graph)
    nodes = list(email_graph.nodes())

    # create the node2vec model
    logger.info("Creating the model")
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    model = Node2Vec(data.edge_index,
                     embedding_dim=embedding_dim,
                     walk_length=walk_length,
                     context_size=context_size,
                     walks_per_node=walks_per_node,
                     num_negative_samples=num_negative_samples,
                     p=p, 
                     q=q,
                     sparse=sparse).to(device)

    # define a dataloader
    logger.info("Creating the data loader")
    loader = model.loader(batch_size=48, shuffle=True, num_workers=0)

    # define optimizer
    logger.info("Creating the optimizer")
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)

    # define training loop
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader,disable=not show_progress):
            print(pos_rw.shape)
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
        return total_loss / len(loader)

    @torch.no_grad()
    def get_embedding():
        model.eval()
        embedding = model()
        return embedding

    # start training
    logger.info("Started training")
    model_queue = deque(maxlen=patience)
    early_stopper = EarlyStopper(stopping_delta=early_stopping_delta,patience=patience)
    for epoch in range(0, num_epochs):
        loss = train()

        # log the params
        print(f'Epoch: {epoch:02d} / Loss: {loss:.4f} / Best Loss : {early_stopper.best_loss:.4f} / Change delta : {early_stopper.change: .4f} ')
        
        if early_stopper.should_stop(loss,epoch):
            logger.warning(f"Early stopping at: {epoch}")
            break
        else:
            model_queue.append(copy.deepcopy(model))    

        # reset the invariants
        loss = 0.0

    # get the embeddings
    model = model_queue.popleft()
    embeddings = get_embedding().detach().cpu()
    logger.info(f"Shape of embeddings : {embeddings.shape}")

    # plot
    node_tensor = torch.Tensor(list(nodes)).int()
    plotter.plot_and_save_embeddings(embeddings,node_tensor,node_labels,"results/figures/email_nodes_geo.png")

def test():
    build()


