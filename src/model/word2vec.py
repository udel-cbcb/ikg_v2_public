from platform import node
import torch.nn as nn
import torch
EPS = 1e-15
import torch.nn.functional as F

class Word2VecSkipGram(nn.Module):
    
    def __init__(self,
                num_nodes,
                embedding_dim,
                padding_index
                ):

        super(Word2VecSkipGram, self).__init__()
        
        # params
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes+1                
        self.padding_index = padding_index
        self.target_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.context_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)

        # set padding index to a very small value
        with torch.no_grad():
            self.target_embedding.weight[self.padding_index] = 0
            self.context_embedding.weight[self.padding_index] = 0

        # init embeddings
        nn.init.uniform_(self.context_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.target_embedding.weight.data,a=-1,b=1)
                
    def forward(self,target_nodes,context_nodes_pos,context_nodes_neg):
        target_emb = self.target_embedding(target_nodes).unsqueeze(1)
        
        context_emb = self.context_embedding(context_nodes_pos)
        context_emb_neg = self.context_embedding(context_nodes_neg)

        prod_pos = torch.mul(target_emb,context_emb)
        pos_sum = torch.sum(prod_pos,dim=1) + EPS
        pos_score = -F.logsigmoid(pos_sum)

        prod_neg = torch.mul(target_emb,context_emb_neg)
        neg_sum = torch.sum(prod_neg,dim=1) + EPS
        neg_score = -F.logsigmoid(1-neg_sum)

        return torch.mean(pos_score + neg_score)

    @torch.no_grad()
    def get_embedding(self,to_cpu=True):
        if to_cpu == True:
            return self.target_embedding.weight.detach().cpu()
        else:
            return self.target_embedding.weight.detach()


class Word2VecCbow(nn.Module):
    
    def __init__(self,
                num_nodes,
                embedding_dim,
                padding_index
                ):

        super(Word2VecCbow, self).__init__()
        
        # params
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes + 1           
        self.padding_index = padding_index
        self.target_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.context_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)

        # init embeddings
        nn.init.uniform_(self.target_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.context_embedding.weight.data,a=-1,b=1)


        # set padding index to a very small value
        with torch.no_grad():
            self.target_embedding.weight[self.padding_index] = 1
            self.context_embedding.weight[self.padding_index] = 1
    
                 
    def forward(self,pos_nodes,neg_nodes,context_nodes):
        
        # get pos embedding
        pos_emb = self.target_embedding(pos_nodes)

        # get neg embedding
        neg_emb = self.target_embedding(neg_nodes)

        # get context embedding
        context_emb = self.context_embedding(context_nodes).mean(dim=1)

        # product
        pos_product = torch.mul(pos_emb,context_emb)
        neg_product = torch.mul(neg_emb,context_emb)

        # sum
        pos_sum = torch.sum(pos_product,dim=1) + EPS
        neg_sum = torch.sum(neg_product,dim=1) + EPS

        # score
        pos_score = -F.logsigmoid(pos_sum)
        neg_score = -F.logsigmoid(1 - neg_sum)
        
        # final score
        loss = torch.mean(pos_score + neg_score)
        return loss


    @torch.no_grad()
    def get_embedding(self,to_cpu=True):
        if to_cpu == True:
            return self.context_embedding.weight.detach().cpu()
        else:
            return self.context_embedding.weight.detach()


class Word2VecSkipGramTriple(nn.Module):
    
    def __init__(self,
                num_nodes,
                embedding_dim,
                padding_index
                ):

        super(Word2VecSkipGramTriple, self).__init__()
        
        # params
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes+1                
        self.padding_index = padding_index
        self.target_head_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.target_tail_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.target_rel_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)

        self.context_head_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.context_tail_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.context_rel_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)

        # init embeddings
        nn.init.uniform_(self.target_head_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.target_tail_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.target_rel_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.context_head_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.context_tail_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.context_rel_embedding.weight.data,a=-1,b=1)

        # set padding index to a very small value
        with torch.no_grad():
            self.target_head_embedding.weight[self.padding_index] = 0
            self.target_tail_embedding.weight[self.padding_index] = 0
            self.target_rel_embedding.weight[self.padding_index] = 0
            self.context_head_embedding.weight[self.padding_index] = 0
            self.context_tail_embedding.weight[self.padding_index] = 0
            self.context_rel_embedding.weight[self.padding_index] = 0

                
    def forward(self,target_triples,pos_context,neg_context):
        
        # get target head rel and tail
        target_head = target_triples[:,0]
        target_rel = target_triples[:,1]
        target_tail = target_triples[:,2]

        # get pos context embedding
        pos_heads = pos_context[:,:,0] # M x N tensor with N being heads for Mth triple in window
        pos_rels = pos_context[:,:,1]
        pos_tails = pos_context[:,:,2]

        # neg pos context embedding
        neg_heads = neg_context[:,:,0] # M x N tensor with N being heads for Mth triple in window
        neg_rels = neg_context[:,:,1]
        neg_tails = neg_context[:,:,2]

        # final loss
        loss_heads = self.get_loss(target_head,
                                    pos_heads,
                                    neg_heads,
                                    target_embedding=self.target_head_embedding,
                                    context_embedding=self.context_head_embedding)

        loss_tails = self.get_loss(target_tail,
                                    pos_tails,
                                    neg_tails,
                                    target_embedding=self.target_tail_embedding,
                                    context_embedding=self.context_tail_embedding
                                )

        loss_rels = self.get_loss(target_rel,
                                    pos_rels,
                                    neg_rels,
                                    target_embedding=self.target_rel_embedding,
                                    context_embedding=self.context_rel_embedding
                                )

        loss = torch.mean(loss_heads + loss_tails + loss_rels)

        return loss

    def get_loss(self,target_nodes,context_nodes_pos,context_nodes_neg,target_embedding,context_embedding):

        target_emb = target_embedding(target_nodes).unsqueeze(1)
        
        context_emb = context_embedding(context_nodes_pos)
        context_emb_neg = context_embedding(context_nodes_neg)

        prod_pos = torch.mul(target_emb,context_emb)
        pos_sum = torch.sum(prod_pos,dim=1) + EPS
        pos_score = -F.logsigmoid(pos_sum)

        prod_neg = torch.mul(target_emb,context_emb_neg)
        neg_sum = torch.sum(prod_neg,dim=1) + EPS
        neg_score = -F.logsigmoid(1-neg_sum)
        
        return torch.mean(pos_score + neg_score)
        
    @torch.no_grad()
    def get_head_embedding(self,to_cpu=True):
        if to_cpu == True:
            return self.target_head_embedding.weight.detach().cpu()
        else:
            return self.target_head_embedding.weight.detach()

    @torch.no_grad()
    def get_tail_embedding(self,to_cpu=True):
        if to_cpu == True:
            return self.target_tail_embedding.weight.detach().cpu()
        else:
            return self.target_tail_embedding.weight.detach()


class Word2VecSkipGramTripleNoRel(nn.Module):
    
    def __init__(self,
                num_nodes,
                embedding_dim,
                padding_index
                ):

        super(Word2VecSkipGramTripleNoRel, self).__init__()
        
        # params
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes+1                
        self.padding_index = padding_index
        self.target_head_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.target_tail_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)

        self.context_head_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.context_tail_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)

        # init embeddings
        nn.init.uniform_(self.target_head_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.target_tail_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.context_head_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.context_tail_embedding.weight.data,a=-1,b=1)


        # set padding index to a very small value
        with torch.no_grad():
            self.target_head_embedding.weight[self.padding_index] = 0
            self.target_tail_embedding.weight[self.padding_index] = 0
            self.context_head_embedding.weight[self.padding_index] = 0
            self.context_tail_embedding.weight[self.padding_index] = 0

                
    def forward(self,target_triples,pos_context,neg_context):
        
        # get target head rel and tail
        target_head = target_triples[:,0]
        target_tail = target_triples[:,2]

        # get pos context embedding
        pos_heads = pos_context[:,:,0] # M x N tensor with N being heads for Mth triple in window
        pos_tails = pos_context[:,:,2]

        # neg pos context embedding
        neg_heads = neg_context[:,:,0] # M x N tensor with N being heads for Mth triple in window
        neg_tails = neg_context[:,:,2]

        # final loss
        loss_heads = self.get_loss(target_head,
                                    pos_heads,
                                    neg_heads,
                                    target_embedding=self.target_head_embedding,
                                    context_embedding=self.context_head_embedding)

        loss_tails = self.get_loss(target_tail,
                                    pos_tails,
                                    neg_tails,
                                    target_embedding=self.target_tail_embedding,
                                    context_embedding=self.context_tail_embedding
                                )


        loss = torch.mean(loss_heads + loss_tails)

        return loss

    def get_loss(self,target_nodes,context_nodes_pos,context_nodes_neg,target_embedding,context_embedding):

        target_emb = target_embedding(target_nodes).unsqueeze(1)
        
        context_emb = context_embedding(context_nodes_pos)
        context_emb_neg = context_embedding(context_nodes_neg)

        prod_pos = torch.mul(target_emb,context_emb)
        pos_sum = torch.sum(prod_pos,dim=1) + EPS
        pos_score = -F.logsigmoid(pos_sum)

        prod_neg = torch.mul(target_emb,context_emb_neg)
        neg_sum = torch.sum(prod_neg,dim=1) + EPS
        neg_score = -F.logsigmoid(1-neg_sum)
        
        return torch.mean(pos_score + neg_score)
        
    @torch.no_grad()
    def get_head_embedding(self,to_cpu=True):
        if to_cpu == True:
            return self.target_head_embedding.weight.detach().cpu()
        else:
            return self.target_head_embedding.weight.detach()

    @torch.no_grad()
    def get_tail_embedding(self,to_cpu=True):
        if to_cpu == True:
            return self.target_tail_embedding.weight.detach().cpu()
        else:
            return self.target_tail_embedding.weight.detach()


class Word2VecCBOWTriple(nn.Module):
    
    def __init__(self,
                num_nodes,
                embedding_dim,
                padding_index
                ):

        super(Word2VecCBOWTriple, self).__init__()
        
        # params
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes+1                
        self.padding_index = padding_index
        self.target_head_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.target_tail_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.target_rel_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)

        self.context_head_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.context_tail_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)
        self.context_rel_embedding = nn.Embedding(self.num_nodes,self.embedding_dim,padding_idx=self.padding_index)

        # init embeddings
        nn.init.uniform_(self.target_head_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.target_tail_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.target_rel_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.context_head_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.context_tail_embedding.weight.data,a=-1,b=1)
        nn.init.uniform_(self.context_rel_embedding.weight.data,a=-1,b=1)

        # set padding index to a very small value
        with torch.no_grad():
            self.target_head_embedding.weight[self.padding_index] = 0
            self.target_tail_embedding.weight[self.padding_index] = 0
            self.target_rel_embedding.weight[self.padding_index] = 0
            self.context_head_embedding.weight[self.padding_index] = 0
            self.context_tail_embedding.weight[self.padding_index] = 0
            self.context_rel_embedding.weight[self.padding_index] = 0

                
    def forward(self,pos_triples,neg_triples,context_triples):
        
        # get pos head, rel and tail
        pos_head = pos_triples[:,0]
        pos_rel = pos_triples[:,1]
        pos_tail = pos_triples[:,2]

        # get neg head, rel and tail
        neg_head = neg_triples[:,0]
        neg_rel = neg_triples[:,1]
        neg_tail = neg_triples[:,2]

        # get context head, rel and tails
        context_head = context_triples[:,:,0] # M x N tensor with N being heads for Mth triple in window
        context_rel = context_triples[:,:,1]
        context_tail = context_triples[:,:,2]


        # final loss
        loss_heads = self.get_loss(pos_head,
                                    neg_head,
                                    context_head,
                                    target_embedding=self.target_head_embedding,
                                    context_embedding=self.context_head_embedding)

        loss_tails = self.get_loss(pos_tail,
                                    neg_tail,
                                    context_tail,
                                    target_embedding=self.target_tail_embedding,
                                    context_embedding=self.context_tail_embedding
                                )

        loss_rels = self.get_loss(pos_rel,
                                    neg_rel,
                                    context_rel,
                                    target_embedding=self.target_rel_embedding,
                                    context_embedding=self.context_rel_embedding
                                )

        loss = torch.mean(loss_heads + loss_tails + loss_rels)

        return loss

    def get_loss(self,pos_nodes,neg_nodes,context_nodes,target_embedding,context_embedding):

        # get pos embedding
        pos_emb = target_embedding(pos_nodes)

        # get neg embedding
        neg_emb = target_embedding(neg_nodes)

        # get context embedding
        context_emb = context_embedding(context_nodes).mean(dim=1)

        # product
        pos_product = torch.mul(pos_emb,context_emb)
        neg_product = torch.mul(neg_emb,context_emb)

        # sum
        pos_sum = torch.sum(pos_product,dim=1) + EPS
        neg_sum = torch.sum(neg_product,dim=1) + EPS

        # score
        pos_score = -F.logsigmoid(pos_sum)
        neg_score = -F.logsigmoid(1 - neg_sum)
        
        # final score
        loss = torch.mean(pos_score + neg_score)
        return loss
        
    @torch.no_grad()
    def get_head_embedding(self,to_cpu=True):
        if to_cpu == True:
            return self.context_head_embedding.weight.detach().cpu()
        else:
            return self.context_head_embedding.weight.detach()

    @torch.no_grad()
    def get_tail_embedding(self,to_cpu=True):
        if to_cpu == True:
            return self.context_tail_embedding.weight.detach().cpu()
        else:
            return self.context_tail_embedding.weight.detach()