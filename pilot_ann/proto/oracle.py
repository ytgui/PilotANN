import torch
import random
from .basic import IndexNSW


class IndexOracle(IndexNSW):
    def __init__(self,
                 d_model: int,
                 known_portion: float,
                 method: str = 'nsw32'):
        IndexNSW.__init__(
            self, d_model=d_model, method=method
        )
        # known
        assert 0.0 < known_portion <= 1.0
        self.known_portion = known_portion

    def entry(self,
              query: torch.Tensor,
              ef_search: int):
        assert query.dim() == 1

        # bruteforce
        dist = torch.cdist(
            query.unsqueeze(0), self.storage, p=2.0
        )
        orders = torch.topk(
            dist, k=ef_search, largest=False, sorted=False
        )
        indices = orders.indices.flatten().tolist()

        #
        nodes = random.choices(
            indices, k=round(self.known_portion * ef_search)
        )
        return nodes
