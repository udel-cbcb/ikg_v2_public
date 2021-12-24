import torch.nn as nn
import torch

class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim,regul_rate):
        super(ComplEx, self).__init__()

        # vars
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = embedding_dim
        self.regul_rate = regul_rate

        # embeddings
        self.ent_re_embeddings = nn.Embedding(self.num_entities, self.dim)
        self.ent_im_embeddings = nn.Embedding(self.num_entities, self.dim)
        self.rel_re_embeddings = nn.Embedding(self.num_relations, self.dim)
        self.rel_im_embeddings = nn.Embedding(self.num_relations, self.dim)

        # init embeddings
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

        # loss criterions
        self.criterion = nn.Softplus()

    def calc_score(self, triples):
        
        # get indices
        h = triples[:,0]
        r = triples[:,1]
        t = triples[:,2]

        # get embeddings
        h_re = self.ent_re_embeddings(h)
        h_im = self.ent_im_embeddings(h)
        t_re = self.ent_re_embeddings(t)
        t_im = self.ent_im_embeddings(t)
        r_re = self.rel_re_embeddings(r)
        r_im = self.rel_im_embeddings(r)
        
        # score
        score = torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            dim=1
        ) 
        
        return score

    def forward(self,positive_triples, negative_triples):
        pos_distance = self.calc_score(positive_triples)
        neg_distance = self.calc_score(negative_triples)

        loss = self._loss(pos_distance,neg_distance)

        if self.regul_rate != 0:
            data = torch.cat((positive_triples,negative_triples))
            loss += self.regul_rate * self.regularization(data)

        return self.loss(pos_distance,neg_distance)

    def _loss(self,pos_score,neg_score):
        loss = (self.criterion(-pos_score).mean() + self.criterion(neg_score).mean()) / 2
        return loss

    def regularization(self, triples):
        # get indices
        h = triples[:,0]
        r = triples[:,1]
        t = triples[:,2]

        # get embeddings
        h_re = self.ent_re_embeddings(h)
        h_im = self.ent_im_embeddings(h)
        t_re = self.ent_re_embeddings(t)
        t_im = self.ent_im_embeddings(t)
        r_re = self.rel_re_embeddings(r)
        r_im = self.rel_im_embeddings(r)
        regul = (torch.mean(h_re ** 2) + 
                 torch.mean(h_im ** 2) + 
                 torch.mean(t_re ** 2) +
                 torch.mean(t_im ** 2) +
                 torch.mean(r_re ** 2) +
                 torch.mean(r_im ** 2)) / 6
        return regul

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()