import torch
import torch.nn as nn
import torch.nn.functional as F

class DistMult(nn.Module):

    def __init__(self,
            num_entities,
            num_relations,
            embedding_dim,
            l1_regul_rate,  
        ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.l1_regul_rate = l1_regul_rate

        # embeddings        
        self.ent_embedding = nn.Embedding(self.num_entities,embedding_dim=self.embedding_dim)
        self.rel_embedding = nn.Embedding(self.num_relations,embedding_dim=self.embedding_dim)

        # init embeddings
        nn.init.uniform_(self.ent_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.rel_embedding.weight.data,a=-1,b=1)

        # loss criterions
        self.criterion = nn.Softplus()

    def forward(self,positive_triples, negative_triples):
        pos_distance = self.calc_score(positive_triples)
        neg_distance = self.calc_score(negative_triples)

        loss = self._loss(pos_distance,neg_distance)

        if self.l1_regul_rate != 0:
            data = torch.cat((positive_triples,negative_triples))
            loss += self.l1_regul_rate * self.regularization(data)
        
        return loss
        
    def _loss(self, pos_score, neg_score):
        loss_value = (self.criterion(-pos_score).mean() + self.criterion(neg_score).mean()) / 2
        return loss_value

    def calc_score(self,triples):
        # get indices
        h = triples[:,0]
        r = triples[:,1]
        t = triples[:,2]

        # get embeddings
        h_emb = self.ent_embedding(h)
        r_emb = self.rel_embedding(r)
        t_emb = self.ent_embedding(t)

        score = (h_emb * r_emb) * t_emb

        score = torch.sum(score,dim=1)

        return score

    def regularization(self, triples):
		# get indices
        h = triples[:,0]
        r = triples[:,1]
        t = triples[:,2]
        
        # get embeddings
        h_emb = self.ent_embedding(h)
        r_emb = self.rel_embedding(r)
        t_emb = self.ent_embedding(t)
        
        regul = (torch.mean(h_emb ** 2) + torch.mean(t_emb ** 2) + torch.mean(r_emb ** 2)) / 3
        
        return regul
    
    @torch.no_grad()
    def get_ent_embedding(self):
        return self.ent_embedding.weight.detach()

    @torch.no_grad()
    def get_rel_embedding(self):
        return self.rel_embedding.weight.detach()

    @torch.no_grad()
    def predict(self,triple):
        
        # pos scores will be closer to 1 and negative scores will be closer to 0
        scores = self.calc_score(triple)
        probs = F.sigmoid(scores)

        return probs
