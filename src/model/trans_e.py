import torch
import torch.nn as nn
from torch.nn.functional import sigmoid

class TransE(nn.Module):

    def __init__(self,
            num_entities,
            num_relations,
            embedding_dim,
            margin=1.0,
            norm=1    
        ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.norm = norm
        self.embedding_dim = embedding_dim

        # embeddings        
        self.ent_embedding = nn.Embedding(self.num_entities,embedding_dim=self.embedding_dim)
        self.rel_embedding = nn.Embedding(self.num_relations,embedding_dim=self.embedding_dim)

        # init embeddings
        nn.init.uniform_(self.ent_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.rel_embedding.weight.data,a=-1,b=1)

        # loss criterions
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self,positive_triples, negative_triples):
        pos_distance = self.calc_distance(positive_triples)
        neg_distance = self.calc_distance(negative_triples)

        return self.loss(pos_distance,neg_distance)
        
    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=positive_distances.device)
        return self.criterion(positive_distances, negative_distances, target).mean()

    def calc_distance(self,triples):
        # get indices
        h = triples[:,0]
        r = triples[:,1]
        t = triples[:,2]

        # get embeddings
        h_emb = self.ent_embedding(h)
        r_emb = self.rel_embedding(r)
        t_emb = self.ent_embedding(t)
                
        distance = ((h_emb + r_emb) - t_emb).norm(p=self.norm,dim=1) 

        return distance
    
    @torch.no_grad()
    def get_ent_embedding(self):
        return self.ent_embedding.weight.detach()

    @torch.no_grad()
    def get_rel_embedding(self):
        return self.rel_embedding.weight.detach()

