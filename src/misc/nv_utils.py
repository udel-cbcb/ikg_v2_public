from torch.utils.data.dataset import Dataset, TensorDataset
from torch_rw import utils as rw_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch_rw import rw

def generate_walks(graph,p,q,walk_length,walks_per_node,batch_size,device,seed):
    
    # to csr
    row_ptr, col_idx = rw_utils.to_csr(graph)
    original_nodes = rw_utils.nodes_tensor(graph)
    nodes = original_nodes.repeat_interleave(walks_per_node)

    row_ptr = row_ptr.to(device)
    col_idx = col_idx.to(device)    

    # create the data set
    dataset = TensorDataset(nodes)

    # create data loader
    dataloader = DataLoader(dataset=dataset,shuffle=False,batch_size=batch_size)

    # perform walks
    walks_list = []
    for data in tqdm(dataloader,total=len(dataloader)):
        nodes = data[0].to(device)
        walks = rw.walk(row_ptr=row_ptr,
                               col_idx=col_idx,
                               target_nodes=nodes,
                               p=p,
                               q=q,
                               walk_length=int(walk_length),
                               seed=seed)
        walks_list.append(walks)

    walks_tensor = torch.cat(walks_list,0)

    assert walks_tensor.shape[1] == walk_length + 1

    return walks_tensor

def to_context(walks,context_size,num_nodes):
    target_nodes_tensor, pos_context_tensor,neg_context_tensor = rw.to_windows(walks,context_size,num_nodes,20)
    return target_nodes_tensor,pos_context_tensor,neg_context_tensor

def generate_contexts(walks_dataset,context_size,num_negatives,num_nodes):
    dataloader = DataLoader(dataset=walks_dataset,batch_size=128)
    pos_context_list = []
    neg_contxt_list = []
    target_nodes_list = []
    for walks in tqdm(dataloader,total=len(dataloader)):
        walks = walks.repeat_interleave(num_negatives,0)
        target_node,pos_context,neg_context = to_context(walks=walks,
                                                        context_size=context_size,
                                                        num_nodes=num_nodes
                                                    )
        target_nodes_list.append(target_node)
        pos_context_list.append(pos_context)
        neg_contxt_list.append(neg_context)

    target_node_tensor = torch.cat(target_nodes_list,0)
    pos_context_tensor = torch.cat(pos_context_list,0)
    neg_context_tensor = torch.cat(neg_contxt_list,0)

    return target_node_tensor,pos_context_tensor,neg_context_tensor


class WalksDataSet(Dataset):

    def __init__(self,walks) -> None:
        super().__init__()
        self.walks = walks

    def __len__(self):
        return len(self.walks)

    def __getitem__(self, index):
        return self.walks[index]