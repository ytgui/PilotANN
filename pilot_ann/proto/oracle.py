import torch
import random
from pilot_ann import proto


class IndexOracle(proto.IndexNSW):
    def __init__(self,
                 d_model: int,
                 known_portion: float,
                 method: str = 'nsw32'):
        proto.IndexNSW.__init__(
            self, d_model=d_model, method=method
        )
        # known
        assert 0.0 < known_portion <= 1.0
        self.known_portion = known_portion

    def entry(self,
              query: torch.Tensor,
              ef_search: int):
        assert query.dim() == 2

        # brute
        dist = torch.cdist(
            query, self.storage, p=2.0
        )
        orders = torch.topk(
            dist, k=ef_search, dim=-1, largest=False
        )

        # entry
        n = round(self.known_portion * ef_search)
        nodes = [
            random.choices(indices, k=n)
            for indices in orders.indices.tolist()
        ]
        return nodes
