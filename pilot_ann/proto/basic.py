import torch
import random
from torch import nn
from pilot_ann import utils


class IndexNSW(nn.Module):
    def __init__(self,
                 d_model: int,
                 method: str = 'nsw32'):
        nn.Module.__init__(self)
        # builder
        self.method = method
        self.d_model = d_model
        for graph_type in ['nsw', 'nsg']:
            if method.startswith(graph_type):
                self.graph_type = graph_type
                self.n_neighbors = int(
                    method.removeprefix(graph_type)
                )
                break
        # storage
        self.indptr: list
        self.indices: list
        self.mapping: list
        self.storage: torch.Tensor

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
        self.indptr, self.indices = indptr, indices
        self.mapping = list(range(x.size(0)))
        self.register_buffer('storage', x)

    def entry(self,
              query: torch.Tensor,
              ef_search: int):
        assert query.dim() == 1

        # nodes
        nodes = set()
        while len(nodes) < ef_search:
            u = random.choice(self.indices)
            if u < 0:
                continue
            nodes.add(u)
        return list(nodes)

    def search(self,
               query: torch.FloatTensor,
               k: int, ef_search: int):
        assert query.dim() == 2
        batch_size = query.size(0)

        # traverse
        output = [
            utils.search_simple(
                indptr=self.indptr,
                indices=self.indices,
                mapping=self.mapping,
                storage=self.storage,
                query=query[i],
                initial=self.entry(
                    query=query[i],
                    ef_search=ef_search
                ),
                k=k, ef_search=ef_search
            )
            for i in range(batch_size)
        ]

        # output
        topk = torch.LongTensor(
            [item['topk'] for item in output]
        )
        n_steps = sum(
            item['n_steps'] for item in output
        )
        n_visited = sum(
            item['n_visited'] for item in output
        )
        output = {
            'topk': topk,
            'n_steps': n_steps / len(output),
            'n_visited': n_visited / len(output)
        }

        #
        return output
