import enum
from networkx.classes.function import nodes
from ray.state import node_ids
import torch
import torch.nn as nn
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, PyTorchCallback
import torch
from torch.nn.functional import embedding
from torch.utils.data.dataset import TensorDataset
from src.model.utils import EarlyStopper
from src.model.word2vec import Word2VecCBOWTriple
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
from src.callbacks.hetero_embedding.triple_walk_embedder_callback import TripleWalkCallBack
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from determined.pytorch import Reducer as reducer
import src.misc.utils as misc_utils
import random

class TripleWalkEmbedderCbow(PyTorchTrial):
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
        self.train_triples_tensor = data["train_triples_tensor"]
        self.relation_tail_idx = data["relation_tail_index"] 
        self.entities_tensor = data["entities_tensor"]
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
        self.w2vmodel = Word2VecCBOWTriple(num_nodes=len(self.entities_tensor),
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
        train_edges_tensor = torch.from_numpy(train_triples_indexed_pd.to_numpy())
                
        # read the entity mapping
        train_entity_mapping_pd = pd.read_csv(f"{fold_path_antd}/train_entity_map.csv")
        train_entity_mapping_pd = train_entity_mapping_pd.sort_values(by="index",ascending=True)

        # read the relation mapping
        train_relation_mapping_pd = pd.read_csv(f"{fold_path_antd}/train_relation_map.csv")
        train_relation_mapping_pd = train_relation_mapping_pd.sort_values(by="index",ascending=True)

        # convert to dict with entity:index pairs
        logger.info(f"Creating node index mapping dict")
        entity_id_dict = train_entity_mapping_pd.set_index("entity").to_dict()['index']

        # nodes tensor
        entities_list = train_entity_mapping_pd["index"].to_list()
        relation_list = train_relation_mapping_pd["index"].to_list()
        all_entities_tensor,_ = torch.Tensor(entities_list+relation_list).to(int).sort(descending=False)
        nodes_tensor,_ = torch.Tensor(entities_list).to(int).sort(descending=False)

        # build node edge index
        logger.info("Creating node edge index mapping")
        relation_tail_index, train_triples_sorted = rw_utils.build_relation_tail_index(train_edges_tensor,all_entities_tensor)

        # create the padding index
        padding_index = all_entities_tensor[-1] + 1
        logger.info(f"Padding idx is : {padding_index}")

        # create target nodes tensor
        logger.info("Creating target nodes tensor")
        sample_targets = self.data_config["sample_targets"]
        proteins = pd.read_csv(f"{data_dir}/processed/proteins.csv")["protein"].tolist()
        protein_idxs = []

        for protein in proteins:
            idx = entity_id_dict[protein]
            protein_idxs.append(idx)

        if sample_targets == True:            
            # sample targets
            sample_idx = np.random.randint(low=0, high=len(nodes_tensor), size=int(len(nodes_tensor)*0.20))
            sampled_entities = nodes_tensor[sample_idx].tolist()
            sampled_entities.extend(protein_idxs) # add proteins to sampled entities
            sampled_entities = list(set(sampled_entities)) # remove duplicates

            target_nodes_tensor = torch.Tensor(sampled_entities).to(int)
        else:
            target_nodes_tensor = nodes_tensor    

        logger.info(f"Number of target nodes: {len(target_nodes_tensor)}")

        protein_idx_tensor = torch.Tensor(protein_idxs).to(int)

        # move to gpu
        train_triples_sorted = train_triples_sorted.cuda()
        relation_tail_index = relation_tail_index.cuda()
        all_entities_tensor = all_entities_tensor.cuda()
        target_nodes_tensor = target_nodes_tensor.cuda()

        data = {
            "train_triples_tensor": train_triples_sorted,
            "relation_tail_index": relation_tail_index,
            "entities_tensor": all_entities_tensor,
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
        neg_edges_val = pd.read_csv(f"{data_dir}/processed/edges/fold_{fold_n}/negative_edges_val.csv").drop(columns=["label"])
        pos_edges_val = pd.read_csv(f"{data_dir}/processed/edges/fold_{fold_n}/positive_edges_val.csv").drop(columns=["label"])

        # map entities to their ids
        logger.info("Mapping edges to entity ids")
        pos_edges_train = self._to_indexed(pos_edges_train,entity_id_dict)
        neg_edges_train = self._to_indexed(neg_edges_train,entity_id_dict)

        pos_edges_val = self._to_indexed(pos_edges_val,entity_id_dict)
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
        return {"callback": TripleWalkCallBack(trial = self,
                                        early_stopping_delta=self.hparams["early_stopping_delta"],
                                        early_stopping_patience=self.hparams["early_stopping_patience"],
                                        protein_idx=self.protein_idx,
                                        fold_n=fold_n,
                                        protein_labels = self.protein_labels,
                                        model_type="cbow"
                                    )}

    def train_batch(self, batch, epoch_idx: int, batch_idx: int):
        #logger.info(f"Epoch: {epoch_idx} / Batch idx: {batch_idx} / Batch len: {len(batch[0])}")
        pos_triples = batch[0]
        neg_triples = batch[1]
        context_triples = batch[2]

        # get the loss
        loss = self.model(pos_triples,neg_triples,context_triples)

        # backward pass
        self.context.backward(loss)
        
        # step the optimizer
        self.context.step_optimizer(self.optimizer,
                                    clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(params, 0.5, error_if_nonfinite=True))

        return {"train_loss":loss}

    def evaluate_batch(self, batch,batch_idx):
        pos_triples = batch[0]
        neg_triples = batch[1]
        context_triples = batch[2]

        # get the loss
        loss = self.model(pos_triples,neg_triples,context_triples)

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

        # get edge operator
        edge_op = self.hparams["edge_op"]

        # get embeddings
        head_embeddings = self.model.get_head_embedding(to_cpu=False)
        tail_embeddings = self.model.get_tail_embedding(to_cpu=False)

        # get edge embeddings train
        logger.info("Creating edge embeddings train")
        head_embedding_train = head_embeddings[train_edges[:,0]]
        tail_embedding_train = tail_embeddings[train_edges[:,1]]
        edge_embedding_train = misc_utils.build_edge_tensor(head_embedding_train,tail_embedding_train,edge_op)

        # get edge embeddings val
        logger.info("Creating edge embeddings val")
        head_embedding_val = head_embeddings[val_edges[:,0]]
        tail_embedding_val = tail_embeddings[val_edges[:,1]]
        edge_embedding_val = misc_utils.build_edge_tensor(head_embedding_val,tail_embedding_val,edge_op)

        # move embedding to cpu and convert to numpy
        edge_embedding_train = edge_embedding_train.cpu().numpy()
        edge_embedding_val = edge_embedding_val.cpu().numpy()


        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=self.seed).fit(edge_embedding_train,train_labels)
        y_predicted_proba = clf.predict_proba(edge_embedding_val)

        # calculate avg precision
        from sklearn.metrics import roc_auc_score
        au_roc = roc_auc_score(val_labels,y_predicted_proba[:,1])

        return au_roc

    def evaluate_au_roc_clf(self):

        node_indices = self.labelled_data_ec["node_indices"]
        labels = self.labelled_data_ec["labels"].numpy()

        # get embeddings
        head_embeddings = self.model.get_head_embedding(to_cpu=False)
        tail_embeddings = self.model.get_tail_embedding(to_cpu=False)
        embeddings = (head_embeddings + tail_embeddings)/2.0
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

        # repeat nodes according to walks per node
        nodes_batch = nodes_batch.repeat_interleave(self.hparams["walks_per_node"],0)

        # shuffle the nodes
        idx = torch.randperm(nodes_batch.nelement())
        nodes_batch = nodes_batch.view(-1)[idx].view(nodes_batch.size()).contiguous()

        # generate walks
        walks = rw.walk_triples(
            self.train_triples_tensor,
            self.relation_tail_idx,
            target_nodes=nodes_batch,
            padding_idx=self.padding_idx,
            walk_length=self.hparams["walk_length"],
            seed=self.seed
        )

        # repeat walks
        walks = walks.repeat_interleave(self.hparams["num_negatives"],0)

        #print(walks)

        # generate contexts
        pos_triples,neg_triples,context = rw.to_windows_triples_cbow(walks=walks,
                                                        window_size=self.hparams["context_size"],
                                                        num_nodes=len(self.entities_tensor),
                                                        padding_idx=self.padding_idx,
                                                        triples=self.train_triples_tensor,
                                                        seed=self.seed
                                                    )

        return pos_triples,neg_triples,context

    
    
    def _create_splits(self,nodes):
        nodes_dataset = TensorDataset(nodes)
        train_length = int(len(nodes_dataset) * 0.8)
        val_length = len(nodes_dataset) - train_length
        logger.info(f"Train len: {train_length}")
        train_dataset,val_dataset = random_split(dataset=nodes_dataset,
                                                lengths=[train_length,val_length])

        return train_dataset,val_dataset


    


    





