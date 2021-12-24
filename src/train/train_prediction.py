from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch
from loguru import logger
import networkx as nx
from tqdm import tqdm
import data.data_utils as du
import data.constants as constants
from model.prediction_model import PredictionModel
import torch.nn as nn
from collections import deque
from model.utils import EarlyStopper
import copy

def train(
    ikg_file,
    emb_file,
    exp_file,
    learning_rate,
    batch_size,
    num_epochs,
    patience,
    early_stopping_delta,
    device
):
    
    torch.manual_seed(20)

    # load the ikg data
    logger.info(f"Loading IKG file from: {ikg_file}")
    ikg_pd = pd.read_csv(ikg_file)
    ikg_pd = ikg_pd[ikg_pd["relation"] == constants.PHOSPHORYLATES]

    # load exp data
    logger.info(f"Loading experimental relations") 
    exp_relation_pd = pd.read_csv(exp_file)

    # load embedding
    embedding = pd.read_csv(emb_file)

    # to networkx
    ikg_graph = nx.Graph()
    for _,row in tqdm(ikg_pd.iterrows(),total=len(ikg_pd)):
        head = row["head"]
        tail = row["tail"]

        ikg_graph.add_edge(head,tail)

    logger.info(f"Number of nodes: {len(ikg_graph.nodes())}")
    logger.info(f"Number of edges: {len(ikg_graph.edges())}")

    # subset embeddings
    proteins = list(ikg_graph.nodes())
    embedding = embedding[embedding["node"].isin(proteins)]

    # create index mapping
    proteins = embedding["node"].tolist()

    # create node to index mapping
    logger.info(f"Creating entity index mapping")
    entity_index_mapping = {}
    for node in tqdm(proteins):
        node_index = proteins.index(node)
        entity_index_mapping[node] = node_index
            
    # experimental relations
    exp_rel = []
    for _,row in exp_relation_pd.iterrows():
        head = row["head"]
        tail = row["tail"]
        exp_rel.append((head,tail))

    # remove the exp relations from ikg_graph
    edges_before_removing = len(ikg_graph.nodes())
    ikg_graph.remove_edges_from(exp_rel)
    edges_after_removing = len(ikg_graph.nodes())
    assert edges_before_removing == edges_after_removing, "Edges not removed from graph"      

    # create splits
    ikg_edges_df = pd.DataFrame(data=ikg_graph.edges())
    exp_relation_pd = pd.DataFrame(data=exp_rel)
    train_edges, val_edges, test_edges = du.create_train_val_test_splits(ikg_df=ikg_edges_df,
                                                                               experimental_relations_df=exp_relation_pd)


    # to indexed
    train_edges = _to_indexed(train_edges,entity_index_mapping)
    val_edges = _to_indexed(val_edges,entity_index_mapping)
    test_edges = _to_indexed(test_edges,entity_index_mapping)

    # train
    train_edges,train_labels = _create_tensors_and_labels(train_edges,len(proteins))
    val_edges,val_labels = _create_tensors_and_labels(val_edges,len(proteins))
    test_edges,test_labels = _create_tensors_and_labels(test_edges,len(proteins))

    print(len(train_edges))
    print(len(val_edges))

    # create embedding
    embedding_tensor = torch.Tensor(embedding.drop(columns=["node"]).to_numpy())

    # scale
    from sklearn.preprocessing import StandardScaler
    embedding_tensor = torch.Tensor(StandardScaler().fit_transform(embedding_tensor.numpy())).to(device)

    # create model
    input_dim = embedding_tensor.shape[1]
    model = PredictionModel(input_dim=input_dim)
    model = model.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # define data loader
    train_dataset = TensorDataset(train_edges,train_labels)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

    val_dataset = TensorDataset(val_edges,val_labels)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=True)

    # define loss fcn
    loss_fcn = nn.BCEWithLogitsLoss()

     # tensor board
    writer = SummaryWriter()
    model_queue = deque(maxlen=patience)
    early_stopper = EarlyStopper(early_stopping_delta=early_stopping_delta,patience=patience)
    for epoch in tqdm(range(num_epochs),total=num_epochs):
        train_loss = _train_model(model,optimizer,train_dataloader,embedding_tensor,loss_fcn,device)
        val_loss = _calc_val_loss(model,val_dataloader,embedding_tensor,loss_fcn,device)
        acc = _predict(model,val_dataloader,embedding_tensor,device)

        early_stopper.update(val_loss,epoch)

        logger.info(f"Epoch: {epoch} / Train Loss: {train_loss} / Val Loss: {val_loss} / Acc: {acc}")
        writer.add_scalars("Prediction/Loss",{"Train": train_loss, "Val": val_loss},epoch)
        writer.add_scalars("Prediction/Acc", {"Val": acc},epoch)

        # save the model queue
        if early_stopper.should_stop(val_loss) == False:
            model_queue.append(copy.deepcopy(model))
        else:
            logger.warning(f"Early stopping at : {epoch}")
            break

    model = model_queue.popleft()
    test_dataset = TensorDataset(test_edges,test_labels)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
    test_roc = _predict(model,test_dataloader,embedding_tensor,device)

    print(test_roc)

def _train_model(model,optimizer,dataloader,embedding,loss_fcn,device):
    total_loss = 0
    model.train()
    for batch in tqdm(dataloader,total=len(dataloader),disable=True):
            # reset optimizer
            optimizer.zero_grad()
            edges = batch[0].to(device)
            labels = batch[1].to(device).unsqueeze(1)

            # get embedding
            u_emb = embedding[edges[:,0]]
            v_emb = embedding[edges[:,1]]

            # forward
            predicted_labels = model(u_emb,v_emb)

            # calc loss
            loss = loss_fcn(predicted_labels,labels)
        
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()

    return total_loss.item()/len(dataloader)

@torch.no_grad()
def _calc_val_loss(model,dataloader,embedding,loss_fcn,device):
    total_loss = 0
    model.eval()
    for batch in tqdm(dataloader,total=len(dataloader),disable=True):
            edges = batch[0].to(device)
            labels = batch[1].to(device).unsqueeze(1)

            # get embedding
            u_emb = embedding[edges[:,0]]
            v_emb = embedding[edges[:,1]]

            # forward
            predicted_labels = model(u_emb,v_emb)

            # calc loss
            loss = loss_fcn(predicted_labels,labels)
            total_loss += loss

    return total_loss.item()/len(dataloader)
    
@torch.no_grad()
def _predict(model,dataloader,embedding,device):
    model.eval()
    predicted_labels_list = []
    actual_labels_list = []
    for batch in tqdm(dataloader,total=len(dataloader),disable=True):
            edges = batch[0].to(device)
            labels = batch[1].to(device).unsqueeze(1)

            # get embedding
            u_emb = embedding[edges[:,0]]
            v_emb = embedding[edges[:,1]]

            # forward
            predicted_labels = model(u_emb,v_emb)

            predicted_labels_list.append(predicted_labels)
            actual_labels_list.append(labels)

    predicted_tensor = torch.cat(predicted_labels_list,dim=0).cpu().numpy()
    import numpy as np
    discretize_v = np.vectorize(du.discretize)
    #predicted_tensor = discretize_v(predicted_tensor)
    actual_tensor = torch.cat(actual_labels_list,dim=0).cpu().numpy()

    from sklearn.metrics import roc_auc_score
    roc = roc_auc_score(actual_tensor,predicted_tensor)

    return roc


def _create_tensors_and_labels(edges,num_nodes):
    pos_edges = torch.Tensor(edges).long()
    neg_edges = du.create_negatives(pos_edges,num_nodes).long()
    all_edges = torch.cat((pos_edges,neg_edges),dim=0).long()
    all_labels = torch.cat((torch.ones_like(pos_edges[:,0]),torch.zeros_like(neg_edges[:,0]))).float()

    # shuffle
    shuffled_indices = torch.randperm(len(all_labels))
    all_edges = all_edges[shuffled_indices]
    all_labels = all_labels[shuffled_indices]

    # get uniques
    _,unique_idx = torch.unique(all_edges,dim=0,return_inverse=True)
    all_edges = all_edges[unique_idx]
    all_labels = all_labels[unique_idx]

    return all_edges,all_labels

def _to_indexed(edges,entity_mapping):
    edges_indexed = []
    for edge in edges:
        u_index = entity_mapping[edge[0]]
        v_index = entity_mapping[edge[1]]
        edges_indexed.append((u_index,v_index))

    return edges_indexed

def test():
    train(
        ikg_file="data/processed/kg/ikg.csv",
        emb_file="results/node2vec/embeddings.csv",
        exp_file="data/processed/kg/experimental_triples.csv",
        batch_size=128,
        learning_rate=0.0001,
        num_epochs=10000,
        early_stopping_delta=0.0001,
        patience=10,
        device="cuda"
    )