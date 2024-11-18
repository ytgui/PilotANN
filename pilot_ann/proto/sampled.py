import torch
import random
from pilot_ann import utils, proto


class IndexSampled(proto.IndexNSW):
    def __init__(self,
                 d_model: int,
                 sample_ratio: float,
                 method: str = 'nsw32'):
        proto.IndexNSW.__init__(
            self, d_model=d_model, method=method
        )
        #
        self.sample_ratio = sample_ratio

    def train(self,
              x: torch.Tensor,
              indptr: torch.Tensor = None,
              indices: torch.Tensor = None):
        assert x.size(-1) == self.d_model

        # train
        if indptr is None:
            print('build', self.method)
            indptr, indices = utils.graph_init(
                x, graph_type=self.graph_type,
                n_neighbors=self.n_neighbors
            )
            print('sampling', self.sample_ratio)
            subgraph = utils.subgraph_init(
                indptr=indptr, indices=indices,
                storage=x, graph_type=self.graph_type,
                n_samples=round(self.sample_ratio * x.size(0)),
                n_neighbors=self.n_neighbors
            )
            indptr, indices, _, _ = subgraph

        # save
        self.indptr, self.indices = indptr, indices
        self.mapping = list(range(x.size(0)))
        self.register_buffer('storage', x)
