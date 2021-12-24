import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

class PredictionModelTorch(nn.Module):

    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = 256
        self.linear_module = self.get_linear_module()

    def get_linear_module(self):
        modules = nn.ModuleList([
            nn.Linear(self.input_dim,self.hidden_dim,bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim,128),
            nn.Dropout(p=0.09),
            nn.LeakyReLU(),
            nn.Linear(128,1),
        ])
        return nn.Sequential(*modules)

    def forward(self,edge_emb):
        out = self.linear_module(edge_emb)
        return out



        
