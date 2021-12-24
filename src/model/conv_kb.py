import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from numpy.random import RandomState

EPS = 1e-15

class ConvKB(nn.Module):

    def __init__(self,
                num_entities,
                num_relations,
                embedding_dim,
                out_channels,
                kernel_size,
                dropout,
                lmbda
            ):

        super(ConvKB, self).__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lmbda = lmbda

        self.ent_embeddings = nn.Embedding(self.num_entities, self.embedding_dim) 
        self.rel_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)

        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(self.kernel_size, 3), bias=True)
        self.activation = nn.Tanh()
        self.hidden_dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear((embedding_dim - self.kernel_size + 1) * out_channels, 1, bias=True)

        self.criterion = nn.Softplus()
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        nn.init.xavier_uniform_(self.linear.weight.data)
        nn.init.xavier_uniform_(self.conv.weight.data)

    def forward(self,triples,labels):

        h = self.ent_embeddings(triples[:,0])
        r = self.rel_embeddings(triples[:,1])
        t = self.ent_embeddings(triples[:,2])

        score = self._score_hrt(h, r, t)

        # regularization
        l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        for W in self.conv.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.linear.parameters():
            l2_reg = l2_reg + W.norm(2)

        return self.loss(score,l2_reg,labels)

    def _score_hrt(self, h,r,t):

        # Output layer regularization
        # In the code base only the weights of the output layer are used for regularization
        # c.f. https://github.com/daiquocnguyen/ConvKB/blob/73a22bfa672f690e217b5c18536647c7cf5667f1/model.py#L60-L66
        #self.regularize_if_necessary(self.linear.weight, self.linear.bias)

        # Stack to convolution input
        conv_inp = torch.stack([h, r, t], dim=-1).view(-1, 1, self.embedding_dim, 3)

        # Convolution
        conv_out = self.conv(conv_inp).view(-1, (self.embedding_dim - self.kernel_size + 1)* self.out_channels)
        hidden = self.activation(conv_out)

        # Apply dropout, cf. https://github.com/daiquocnguyen/ConvKB/blob/master/model.py#L54-L56
        hidden = self.hidden_dropout(hidden)

        # Linear layer for final scores
        scores = self.linear(hidden)

        return scores

    def loss(self, score, regul,labels):
        # apply smoothing to labels
        #labels = labels + EPS
        logits = self.criterion(score * labels)
        loss = torch.mean(logits) + self.lmbda * regul
        return loss

    @torch.no_grad()
    def predict(self,triples):
        h = self.ent_embeddings(triples[:,0])
        r = self.rel_embeddings(triples[:,1])
        t = self.ent_embeddings(triples[:,2])
        
        # raw scores
        score = self._score_hrt(h, r, t)

        # to probs
        probs = nn.Sigmoid()(score)

        return probs