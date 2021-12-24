from os import walk
from networkx.classes.function import nodes
from pykeen.stoppers import early_stopping

from sklearn.utils import shuffle
from torch import optim
from torch.random import seed
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from data.visualizer import get_color
from model.node2vec import Node2VecDataLoader, Word2VecSkipGram, WalksDataSet, ContextDataSet
import pandas as pd
from loguru import logger
import networkx as nx
from tqdm import tqdm
import torch
import model.node2vec as nv
import data.data_utils as du
import torch
import model.utils as mu
from torch.utils.tensorboard import SummaryWriter
import data.visualizer as vz
from data import constants

def train(
    ikg_file,
    experimental_file,
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
    evaluate_at_batch,
    stop_at_batch,
    learning_rate,
    device,
    enable_tensorboard=False
):

    torch.manual_seed(20)

    # load the ikg data
    logger.info(f"Loading IKG file from: {ikg_file}")
    ikg_pd = pd.read_csv(ikg_file)
    #ikg_pd = ikg_pd[ikg_pd["relation"] == constants.PHOSPHORYLATES]

    # load exp data
    logger.info(f"Loading experimental relations") 
    exp_relation_pd = pd.read_csv(experimental_file)

    # load parent mapping
    parent_mapping_pd = pd.read_csv("data/processed/go/go_parent_mapping.csv")

    # to networkx
    ikg_graph = nx.Graph()
    for _,row in tqdm(ikg_pd.iterrows(),total=len(ikg_pd)):
        head = row["head"]
        tail = row["tail"]

        ikg_graph.add_edge(head,tail)

    logger.info(f"Number of nodes: {len(ikg_graph.nodes())}")
    logger.info(f"Number of edges: {len(ikg_graph.edges())}")

    # create node to index mapping
    logger.info(f"Creating entity index mapping")
    entity_index_mapping = {}
    nodes = list(ikg_graph.nodes())
    node_colors = []
    for node in tqdm(nodes):
        node_index = nodes.index(node)
        entity_index_mapping[node] = node_index

        # get color
        node_colors.append(vz.get_category(node,parent_mapping_pd))

    # relabel the graph
    ikg_graph = nx.convert_node_labels_to_integers(ikg_graph)

    # experimental relations to index
    exp_rel_indexed = []
    for _,row in exp_relation_pd.iterrows():
        head = row["head"]
        tail = row["tail"]
        exp_rel_indexed.append((head,tail))

    # remove the exp relations from ikg_graph
    edges_before_removing = len(ikg_graph.nodes())
    ikg_graph.remove_edges_from(exp_rel_indexed)
    edges_after_removing = len(ikg_graph.nodes())
    assert edges_before_removing == edges_after_removing, "Edges not removed from graph"      

    # create splits
    ikg_edges_df = pd.DataFrame(data=ikg_graph.edges())
    exp_relation_pd = pd.DataFrame(data=exp_rel_indexed)
    train_triples, val_triples, test_tripels = du.create_train_val_test_splits(ikg_df=ikg_edges_df,
                                                                               experimental_relations_df=exp_relation_pd)

    
    if(rw_batch_size <= 0):
        rw_batch_size = len(ikg_graph.nodes())
        
    # perform random walks on the graph
    logger.info("Generating walks")
    walks_tensor = nv.generate_walks(graph=ikg_graph,
                                     p=p,
                                     q=q,
                                     walk_length=walk_length-1,
                                     walks_per_node=walks_per_node,
                                     batch_size=rw_batch_size,
                                     device=device,
                                     seed=20   
                                    )

        
    # create dataset
    walks_dataset = WalksDataSet(walks_tensor)

    # create the model
    model = Word2VecSkipGram(
        embedding_dim=embedding_dim,
        num_nodes=len(ikg_graph.nodes()),
    )

    model = model.to(device)
    
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    logger.info("Starting training")
    # create the walk data loader
    walk_dataloader = DataLoader(dataset=walks_dataset,batch_size=128,shuffle=True)

    # tensor board
    writer = SummaryWriter()

    # train
    early_stopper = mu.EarlyStopper(early_stopping_delta=early_stopping_delta,patience=patience)
    is_early_stopped=False
    is_stopped_at_batch=False
    batch_index = 0
    for _ in tqdm(range(num_epochs),total=num_epochs,disable=False):
        epoch_loss = 0
        for walks in tqdm(walk_dataloader,total=len(walk_dataloader), disable=False):
            
            walks = walks.to(device)
            walks = walks.repeat_interleave(num_negative_samples,0)

            # to context
            target_nodes, pos_context, neg_context = nv.to_context(walks=walks,
                                                                context_size=context_size,
                                                                num_nodes=len(nodes))

            # create context dataset
            context_dataset = ContextDataSet(target_nodes,pos_context,neg_context)
            context_dataloader = DataLoader(dataset=context_dataset,batch_size=128,shuffle=False)

            # perform training
            loss = _train_model(model,optimizer,context_dataloader,device)
            early_stopper.update(loss,batch_index)

            # log
            if enable_tensorboard == True:
                    writer.add_scalar("Loss/IKG",loss,batch_index)

            if batch_index != 0 and batch_index % 20 == 0:
                logger.info(f"Batch index: {batch_index} / Loss: {loss}")
                            
            # stop at this batch if required
            if stop_at_batch != -1 and stop_at_batch == batch_index:
                is_stopped_at_batch = True
                break                  

            # update batch index            
            batch_index = batch_index + 1


        if is_stopped_at_batch == True:
            logger.warning(f"Stopping at batch : {batch_index}")
            break
        
        if is_early_stopped == True:
            logger.warning(f"Early stopping at batch: {batch_index}")
            break

    # save embedding
    logger.info("Saving embeddings")
    embedding = model.get_embedding().numpy()
    writer.add_embedding(embedding,metadata=node_colors)
    embedding_df = pd.DataFrame(data=embedding)
    embedding_df.insert(0,"node",nodes)
    embedding_df.to_csv("results/node2vec/embeddings.csv",index=False)

    logger.info("Done")


def evaluate_model(model,train_edges,val_edges,train_labels,val_labels):
    train_edges_vec = model.get_embedding_for_edges(train_edges).cpu().numpy()
    val_edges_vec = model.get_embedding_for_edges(val_edges).cpu().numpy()
    train_labels = train_labels.cpu().numpy()
    val_labels = val_labels.cpu().numpy()

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=20).fit(train_edges_vec, train_labels)
    labels_predicted = clf.predict(val_edges_vec)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(val_labels, labels_predicted)

    return acc



def _train_model(model,optimizer,dataloader,device):
    total_loss = 0
    #model.train()
    for batch in tqdm(dataloader,total=len(dataloader),disable=True):
            # reset optimizer
            optimizer.zero_grad()
            
            target_nodes = batch[0].to(device)
            pos_context = batch[1].to(device)
            neg_context = batch[2].to(device)

            loss = model(target_nodes,pos_context,neg_context)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach()

    return total_loss.item()/len(dataloader)


def test():
    train(
        ikg_file="data/processed/kg/ikg.csv",
        experimental_file="data/processed/kg/experimental_triples.csv",
        embedding_dim=256,
        walk_length=120,
        walks_per_node=10,
        p=1,
        q=0.7,
        context_size=20,
        num_negative_samples=2,
        rw_batch_size=8096,
        num_epochs=10,
        patience=10,
        early_stopping_delta=0.001,
        learning_rate=0.001,
        evaluate_at_batch=100,
        stop_at_batch=-1,
        device="cuda",
        enable_tensorboard=True
    )

