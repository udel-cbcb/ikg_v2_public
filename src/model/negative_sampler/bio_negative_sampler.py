# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of of Bordes *et al.*."""

import math
from typing import Collection, Optional

import torch

from pykeen.sampling.negative_sampler import NegativeSampler

__all__ = [
    'BioNegativeSampler',
]

LOOKUP = {'h': 0, 'r': 1, 't': 2}



class BioNegativeSampler(NegativeSampler):
    
    def __init__(
        self,
        entity_mapping,
        protein_region_mapping,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.entity_mapping_id_to_label = {v: k for k, v in entity_mapping.items()}
        self.entity_mapping_label_to_id = entity_mapping
        self.protein_region_mapping = protein_region_mapping

    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        if self.num_negs_per_pos > 1:
            positive_batch = positive_batch.repeat_interleave(repeats=self.num_negs_per_pos, dim=0)

        # Bind number of negatives to sample
        num_negs = positive_batch.shape[0]

        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()

        for index in range(0,num_negs):
            row = negative_batch[index]
            head=row[0]
            relation=row[1]
            tail=row[2]

            # if phosph relation
            if relation == 0:
                
                # get labels
                _kinase = self.entity_mapping_id_to_label[int(head)]
                substrate = self.entity_mapping_id_to_label[int(tail)]
                
                # get corruped substrate
                #corrupted_substrate = self.get_corruped_substrate(substrate)

                # get index 
                #corrupted_substrate_id = self.entity_mapping_label_to_id[corrupted_substrate]
                # replace by a random entity
                corrupted_substrate_id = torch.randint(high=self.num_entities -1,size=(1,))[0]
                negative_batch[index] = torch.Tensor([head,relation,corrupted_substrate_id]).to(positive_batch.device)

            else:

                # replace by a random entity
                random_id = torch.randint(high=self.num_entities -1,size=(1,))[0]

                negative_batch[index] = torch.Tensor([head,relation,random_id]).to(positive_batch.device)
                                          

        return negative_batch.view(-1, self.num_negs_per_pos, 3)

    def get_corruped_substrate(self,substrate):
        location = self.protein_region_mapping[self.protein_region_mapping["protein"] == substrate].iloc[0]
        
        # if the protein is extra cellular
        if location["is_extracellular"] == True:
            
            # filter
            filtered_df = self.protein_region_mapping[self.protein_region_mapping["is_extracellular"] == False]

        elif location["is_nucleus"] == True:

            # filter
            filtered_df = self.protein_region_mapping[self.protein_region_mapping["is_nucleus"] == False]

        elif location["is_cytoplasm"] == True:

            # filter
            filtered_df = self.protein_region_mapping[self.protein_region_mapping["is_cytoplasm"] == False]

        # randomly choose one
        random_sample = filtered_df.sample(10,random_state=20).iloc[0]

        return random_sample["protein"]

        
        

