from typing import Optional
import pytorch_lightning as pl
from loguru import logger
import model.node2vec as nv
from model.node2vec import WalksDataSet
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from model.node2vec import Node2VecDataLoader

class Node2VecDataModule(pl.LightningDataModule):
    def __init__(self,graph,
                      batch_size,
                      rw_batch_size,
                      walk_length,
                      p,
                      q,
                      num_negatives,
                      context_size,
                      walks_per_node
                      ):
        super().__init__()
        self.graph = graph
        self.batch_size = batch_size
        self.rw_batch_size = rw_batch_size
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.num_negatives = num_negatives
        self.context_size = context_size
        self.walks_per_node = walks_per_node

        if(self.rw_batch_size <= 0):
            self.rw_batch_size = len(self.graph.nodes())

    def setup(self, stage: Optional[str] = None):
        # perform random walks on the graph
        logger.info(f"Generating walks")
        walks_tensor,node = nv.generate_walks(graph=self.graph,
                                        p=self.p,
                                        q=self.q,
                                        walk_length=self.walk_length-1,
                                        walks_per_node=self.walks_per_node,
                                        batch_size=self.rw_batch_size,
                                        device="cuda",
                                        seed=20   
                                    )

        logger.info(f"Walks shape: {walks_tensor.shape}")

        # create dataset
        walks_dataset = WalksDataSet(walks_tensor)

        # split
        train_length = int(len(walks_dataset) * 0.8)
        val_length = len(walks_dataset) - train_length
        train_dataset,val_dataset = random_split(dataset=walks_dataset,
                                                lengths=[train_length,val_length])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return Node2VecDataLoader(dataset=self.train_dataset,
                                    context_size=self.context_size,
                                    num_nodes=len(self.graph.nodes()),
                                    num_negative_samples=self.num_negatives,
                                    batch_size=self.batch_size,
                                    shuffle=True)

    def val_dataloader(self):
        return Node2VecDataLoader(dataset=self.val_dataset,
                                    context_size=self.context_size,
                                    num_nodes=len(self.graph.nodes()),
                                    num_negative_samples=self.num_negatives,
                                    batch_size=self.batch_size,
                                    shuffle=False)