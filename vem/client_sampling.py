import abc
from typing import List
import numpy as np
from numpy.random import default_rng


class ClientSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self) -> int:
        """sample the index of a client"""


class BinomialSampler(ClientSampler):
    def __init__(self, n_clients, p, seed=1234):
        self.n_clients = n_clients
        self.p = p
        self._rng = default_rng(seed)

    def sample(self) -> List[int]:
        client_indices = np.arange(self.n_clients)
        while True:
            mask = self._rng.uniform(size=self.n_clients) < self.p
            if len(client_indices[mask]) > 1:
                break
        return client_indices[mask]
