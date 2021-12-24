import torch.nn as nn
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
import torch.nn.functional as F
import torch
from torch_geometric.nn.models.autoencoder import GAE

class RGCNAutoEncoder(nn.Module):

    def __init__(self, num_entities,
                       num_relations,
                       embedding_dim,
                       dropout,
                       num_blocks
                    ):

        super().__init__()

        # params
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout
        self.num_blocks = num_blocks

        # layers
        encoder = RGCNEncoder(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout_prob,
            num_blocks=self.num_blocks
        )
        decoder = DistMultDecoder(
            num_relations=self.num_relations,
            embedding_dim=self.embedding_dim
        )

        self.gae = GAE(
            encoder=encoder,
            decoder=decoder
        )

    def forward(self,edge_pos,edge_neg,edge_type):
        # encode
        z = self.gae.encode(edge_pos,edge_type)

        # decode
        pos_out = self.gae.decode(z, edge_pos, edge_type)
        
        # decode negative sample
        neg_out = self.gae.decode(z, edge_neg,edge_type)

        # calc loss
        out = torch.cat([pos_out, neg_out])
        gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
        reg_loss = z.pow(2).mean() + self.gae.decoder.rel_emb.pow(2).mean()
        loss = cross_entropy_loss + 1e-2 * reg_loss

        return loss

    def negative_sampling(self,edge_index,num_nodes):
        # Sample edges by corrupting either the subject or the object of each edge.
        mask_1 = torch.rand(edge_index.size(1)) < 0.5
        mask_2 = ~mask_1

        neg_edge_index = edge_index.clone()
        neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ))
        neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ))
        return neg_edge_index

    @torch.no_grad()
    def get_embedding(self):
        return self.gae.encoder.node_emb.data.detach()


class RGCNEncoder(nn.Module):

    def __init__(self,num_entities, num_relations, embedding_dim,dropout,num_blocks):
        super().__init__()
        
        # params
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        # layers
        self.node_emb = nn.Embedding(num_entities,embedding_dim)
        self.conv1 = RGCNConv(embedding_dim, embedding_dim, num_relations,
                              num_blocks=num_blocks)
        self.conv2 = RGCNConv(embedding_dim, embedding_dim, num_relations,
                              num_blocks=num_blocks)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb.weight)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = (self.node_emb(edge_index[:,0]),self.node_emb(edge_index[:,1]))
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

class DistMultDecoder(nn.Module):
    def __init__(self, num_relations, embedding_dim):
        super().__init__()
        self.rel_emb = nn.Parameter(torch.Tensor(num_relations,embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)

    

    

