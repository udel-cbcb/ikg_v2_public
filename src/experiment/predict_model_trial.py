from sklearn import metrics
import torch
import numpy as np
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
import pandas as pd
from pathlib import Path
from torch.utils.data.dataset import TensorDataset
from src.model.prediction_model import PredictionModelTorch
import torch.nn as nn
from src.callbacks.prediction_callback import PredictionCallBack
from sklearn.preprocessing import StandardScaler

class PredictionModelTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        
        torch.manual_seed(20)
        np.random.seed(20) 

        self.context = context
        self.hparams = self.context.get_hparams()
        
        # load data
        data = self._load_data()
        self.train_dataset = data[0]
        self.val_dataset = data[1]
        self.test_dataset = data[2]
        self.embedding_size = data[3]

        # loss fcn
        self.loss_fcn = nn.BCEWithLogitsLoss()

        # create the model
        model = PredictionModelTorch(self.embedding_size)
        self.model = self.context.wrap_model(model)

        # init optimizer
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"])
        )


    def _load_data(self):

        # get the fold
        fold_n = self.hparams["fold_n"]

        # sampler_type
        sampler_type = self.hparams["sampler_type"]

        # load the edge embeddings
        data_dir = f"{Path.home()}/data/ikg_v2_data/negative_sampling"
        train_embeddings = pd.read_csv(f"{data_dir}/processed/edges/{sampler_type}/fold_{fold_n}/embeddings/all_edge_embeddings_train.csv")
        val_embeddings = pd.read_csv(f"{data_dir}/processed/edges/{sampler_type}/fold_{fold_n}/embeddings/all_edge_embeddings_val.csv")
        test_embeddings = pd.read_csv(f"{data_dir}/processed/edges/{sampler_type}/fold_{fold_n}/embeddings/all_edge_embeddings_test.csv")

        scaler = StandardScaler()

        # to torch
        train_X = scaler.fit_transform(train_embeddings.drop(columns=["label"]).to_numpy())
        train_y = torch.from_numpy(train_embeddings["label"].to_numpy()).float().unsqueeze(-1)
        train_dataset = TensorDataset(torch.from_numpy(train_X).float(),train_y)
        embedding_size = train_X.shape[1]

        val_X = scaler.transform(val_embeddings.drop(columns=["label"]).to_numpy())
        val_y = torch.from_numpy(val_embeddings["label"].to_numpy().ravel()).float().unsqueeze(-1)
        val_dataset = TensorDataset(torch.from_numpy(val_X).float(),val_y)
                
        test_X = scaler.transform(test_embeddings.drop(columns=["label"]).to_numpy())
        test_y = torch.from_numpy(test_embeddings["label"].to_numpy()).float().unsqueeze(1)
        test_dataset = TensorDataset(torch.from_numpy(test_X).float(),test_y)

        return train_dataset,val_dataset,test_dataset,embedding_size

    def build_training_data_loader(self):
        print(len(self.train_dataset))
        return DataLoader(self.train_dataset,
                          batch_size=self.context.get_per_slot_batch_size(),
                          shuffle=True,
                          drop_last=True,
                        )
        

    def build_validation_data_loader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.context.get_per_slot_batch_size(),
                          shuffle=True,
                          drop_last=True,
                        )


    def train_batch(self, batch, epoch_idx: int, batch_idx: int):
        
        batch_X = batch[0]
        batch_y = batch[1]

        # get the loss
        prediction = self.model(batch_X)

        loss = self.loss_fcn(prediction,batch_y)

        # backward pass
        self.context.backward(loss)
        
        # step the optimizer
        self.context.step_optimizer(self.optimizer)

        return {"train_loss":loss}

    def evaluate_batch(self, batch):

        batch_X = batch[0]
        batch_y = batch[1]

        # predict
        prediction = self.model(batch_X)

        # loss
        loss = self.loss_fcn(prediction,batch_y)
        
        metrics = {
            "val_loss": loss
        }

        return metrics

    def build_callbacks(self):
        return {"callback": PredictionCallBack(trial = self,
                                        fold_n=self.hparams["fold_n"],
                                        early_stopping_delta=self.hparams["early_stopping_delta"],
                                        early_stopping_patience=self.hparams["early_stopping_patience"]
                                    )}