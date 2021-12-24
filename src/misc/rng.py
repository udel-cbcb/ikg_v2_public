import ray
import numpy as np

@ray.remote
class RngActor(object):

    def __init__(self,seed):
        self.value = 0
        self.rng = np.random.default_rng(seed)

    def get_random_numbers(self,low,high,size):
        return self.rng.integers(low=low, high=high, size=size)