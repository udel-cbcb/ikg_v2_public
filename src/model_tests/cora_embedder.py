import enum
import torch
import torch.nn as nn
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, PyTorchCallback
import torch
from src.model.utils import EarlyStopper
from src.model.word2vec import Word2VecSkipGram, Word2VecCbow
import pandas as pd
from pathlib import Path
import networkx as nx
from torch_rw import utils as rw_utils
import src.misc.nv_utils as nv_utils
from torch.utils.data import random_split
from determined.pytorch import DataLoader
from loguru import logger
from src.misc import plotter
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torch_rw import rw
from torch.utils.data.dataset import TensorDataset
from determined.pytorch import Reducer as reducer

class CoraEmbedderTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        
        torch.manual_seed(20)

        self.tf_logger = TorchWriter()

        # Initialize the trial class and wrap the models, optimizers, and LR schedulers.
        self.context = context
        self.hparams = self.context.get_hparams()
        
        # init data
        self.row_ptr,self.col_idx,self.nodes_tensor,self.node_labels,self.email_graph = self._load_data()
        
        # create splits
        train_dataset,val_dataset = self._create_splits(self.nodes_tensor)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # init model
        model_type = self.hparams["model_type"]
        if model_type == "sg":
            self.w2vmodel = Word2VecSkipGram(num_nodes=len(self.nodes_tensor),
                                            embedding_dim=self.hparams["embedding_dim"],
                                            padding_index=len(self.nodes_tensor)
                                        )
        elif model_type == "cb":
            self.w2vmodel = Word2VecCbow(num_nodes=len(self.nodes_tensor),
                                            embedding_dim=self.hparams["embedding_dim"],
                                            padding_index=len(self.nodes_tensor)
                                            )
        
        self.model = self.context.wrap_model(self.w2vmodel)

        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"])
        )

    def build_callbacks(self):
        return {"callback": CoraCallBack( trial = self,
                                         nodes=self.nodes_tensor,
                                         node_labels=self.node_labels,
                                         tf_logger=self.tf_logger,
                                         early_stopping_delta=self.hparams["early_stopping_delta"],
                                         early_stopping_patience=self.hparams["early_stopping_patience"]
                                    )}

    def train_batch(self, batch, epoch_idx: int, batch_idx: int):
        model_type = self.hparams["model_type"]

        if model_type == "sg":
            target_nodes = batch[0]
            pos_context = batch[1]
            neg_context = batch[2]
            loss = self.model(target_nodes,pos_context,neg_context)
        elif model_type == "cb":        
            pos_nodes = batch[0]
            neg_nodes = batch[1]
            context = batch[2]
            loss = self.model(pos_nodes,neg_nodes,context)


        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer,
                                    clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(params, 0.0001))

        return {"train_loss":loss}

    def evaluate_batch(self, batch, batch_idx):
        model_type = self.hparams["model_type"]

        if model_type == "sg":
            target_nodes = batch[0]
            pos_context = batch[1]
            neg_context = batch[2]
            loss = self.model(target_nodes,pos_context,neg_context)
        elif model_type == "cb":        
            pos_nodes = batch[0]
            neg_nodes = batch[1]
            context = batch[2]
            loss = self.model(pos_nodes,neg_nodes,context)

        rank = self.context.distributed.get_rank()
        roc_score = 0
        if rank == 0 and batch_idx == 0:
            roc_score = self.evaluate_au_roc()
        else:
            roc_score = 0

        metrics = {
            "val_loss":loss,
            "roc_score": roc_score
        }

        return metrics

    def evaluation_reducer(self):
        reducers = {
            "val_loss": reducer.AVG,
            "roc_score": reducer.MAX
        }

        return reducers

    def evaluate_au_roc(self):
        # get embeddings
        embeddings = self.model.get_embedding().numpy()[:-1]

        # encode labels
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        labels = le.fit(self.node_labels["label"]).transform(self.node_labels["label"])

        # create splits
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(embeddings,
                                                            labels,test_size=0.25,random_state=20,shuffle=True)

        logger.info("Training edge predictor")
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=20).fit(X_train,y_train)

        logger.info("Predicting")
        y_predicted_proba = clf.predict_proba(X_test)

        # calculate avg precision
        from sklearn.metrics import roc_auc_score
        au_roc = roc_auc_score(y_test,y_predicted_proba,multi_class="ovo")

        return au_roc

    def build_training_data_loader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.context.get_per_slot_batch_size(),
                          shuffle=True,
                          collate_fn=self.to_sample,
                          drop_last=True
                        )
        

    def build_validation_data_loader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.context.get_per_slot_batch_size(),
                          collate_fn=self.to_sample
                        )

    def _load_data(self):
        data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"

        # load the data
        edges_raw = pd.read_csv(f"{data_dir}/raw/test/cora/cora.cites",sep='\t',header=None)

        # load the labels
        node_labels_raw = pd.read_csv(f"{data_dir}/raw/test/cora/cora.content",sep='\t',header=None)
        node_labels = pd.DataFrame()
        node_labels["node"] = node_labels_raw.to_numpy()[:,0]
        node_labels["label"] = node_labels_raw.to_numpy()[:,-1]
        
        # create graph
        email_graph = nx.from_edgelist(edges_raw.to_numpy())
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

        print(node_labels.head())

        # get csr representation
        row_ptr, col_idx = rw_utils.to_csr(email_graph)
        nodes_tensor = rw_utils.nodes_tensor(email_graph)

        print(nodes_tensor)
        
        # move to gpu
        row_ptr = row_ptr.to(self.context.device)
        col_idx = col_idx.to(self.context.device)
        nodes_tensor = nodes_tensor.to(self.context.device)
        
        return row_ptr,col_idx,nodes_tensor,node_labels,email_graph

    def _create_splits(self,nodes):
        nodes_dataset = TensorDataset(nodes)
        train_length = int(len(nodes_dataset) * 0.9)
        val_length = len(nodes_dataset) - train_length
        logger.info(f"Train len: {train_length}")
        train_dataset,val_dataset = random_split(dataset=nodes_dataset,
                                                lengths=[train_length,val_length])

        return train_dataset,val_dataset


    def to_sample(self,nodes_batch):
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
        model_type = self.hparams["model_type"]
        if model_type == "sg":
            target_nodes,pos_context,neg_context = rw.to_windows(walks=walks,
                                                            window_size=self.hparams["context_size"],
                                                            num_nodes=len(self.nodes_tensor),
                                                            seed=20
                                                        )
        
            return target_nodes,pos_context,neg_context
        elif model_type == "cb":
            pos_nodes,neg_nodes,context = rw.to_windows_cbow(walks=walks,
                                                            window_size=self.hparams["context_size"],
                                                            num_nodes=len(self.nodes_tensor),
                                                            seed=20
                                                        )
        
            return pos_nodes,neg_nodes,context


class CoraCallBack(PyTorchCallback):

    def __init__(self,trial,nodes,node_labels,tf_logger,early_stopping_delta,early_stopping_patience):
        self.trial = trial
        self.model = trial.model
        self.nodes = nodes
        self.node_labels = node_labels
        self.tf_logger = tf_logger
        self.index = 0
        self.early_stopper = EarlyStopper(
            stopping_delta=early_stopping_delta,
            patience=early_stopping_patience
        )
        super().__init__()

    def on_validation_end(self, metrics) -> None:
       
        # check if need to early stop
        val_loss = metrics["val_loss"]
        should_stop = self.early_stopper.should_stop(val_loss,self.index)
        logger.info(f"Index: {self.index} / Best loss: {self.early_stopper.best_loss} / Change: {self.early_stopper.change} / Waited: {self.early_stopper.num_of_epochs_waited}")

        if should_stop:
            logger.info("Requesting early stopping")

            # plot the fig
            self.plot()

            self.index = self.index + 1

            self.plot()

            # request stop
            self.trial.context.set_stop_requested(True)

        # update index
        self.index = self.index + 1

        if self.index % 20 == 0:
            self.plot()

    def on_checkpoint_end(self, _checkpoint_dir: str) -> None:
        # plot the fig
        self.plot()

        self.index = self.index + 1

        self.plot()

    def plot(self):
        rank = self.trial.context.distributed.get_rank()

        if rank == 0:
            # get embeddings
            embeddings = self.model.get_embedding().numpy()[:-1]

            # plot
            logger.info(f"Plotting embeddings for : {len(embeddings)}")
            fig = plotter.plot_embeddings(embeddings,self.nodes.cpu().numpy(),self.node_labels)
            fig_arr = plotter.fig2array(fig)

            # to mplt
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5,5))
            plt.imshow(fig_arr)
            fig_mp = plt.gcf()
            
            self.tf_logger.writer.add_figure("embedding",fig_mp,global_step=self.index+1)


