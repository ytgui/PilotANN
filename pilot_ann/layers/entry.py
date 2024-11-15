import torch
from torch import nn
from pilot_ann import utils, kernels


class IndexEntry(nn.Module):
    def __init__(self,
                 d_model: int,
                 method: str = 'random32'):
        nn.Module.__init__(self)
        #
        self.method = method
        self.d_model = d_model
        # entry point
        self.entry_nodes: torch.Tensor
        self.entry_vectors: torch.Tensor
        self.route_vectors: torch.Tensor

    def nodes(self):
        nodes = self.entry_nodes.flatten()
        return nodes.tolist()

    def train(self,
              storage: torch.Tensor,
              init_nodes: list = None):
        assert isinstance(storage, torch.Tensor)
        if storage.dim() != 2:
            raise RuntimeError
        if storage.size(-1) != self.d_model:
            raise RuntimeError
        if hasattr(self, 'entry_nodes'):
            return
        print('[INFO] building entry')

        # build
        if self.method.startswith('random'):
            n_entries = int(
                self.method.removeprefix('random')
            )
            entry_nodes = torch.randint(
                high=storage.size(0),
                size=[n_entries], dtype=torch.long
            )
            route_vectors = None
        elif self.method.startswith('router'):
            params = self.method.removeprefix('router')
            n_routers, n_leaves = [
                int(x) for x in params.split('x')
            ]
            route_vectors, entry_nodes = utils.router_init(
                storage=storage, init_nodes=init_nodes,
                n_routers=n_routers, n_leaves=n_leaves
            )
        else:
            raise NotImplementedError
        self.register_buffer('entry_nodes', entry_nodes)
        self.register_buffer('entry_vectors', storage[entry_nodes])
        self.register_buffer('route_vectors', route_vectors)
        return self

    @torch.no_grad()
    def search(self, query: torch.Tensor):
        assert query.dim() == 2
        batch_size = query.size(0)

        # entry
        if self.route_vectors is None:
            assert self.entry_vectors.dim() == 2
            EP_D = torch.square(
                torch.cdist(query, self.entry_vectors, p=2.0)
            )
            EP_I = torch.tile(
                self.entry_nodes.unsqueeze(0), dims=[batch_size, 1]
            )
        else:
            assert query.is_cuda is True
            assert self.entry_vectors.dim() == 3
            router, EP_D = kernels.routing_cuda(
                query, self.route_vectors,
                entry_nodes=self.entry_nodes,
                entry_vectors=self.entry_vectors
            )
            EP_I = torch.index_select(
                self.entry_nodes, dim=0, index=router
            )
        return EP_I, EP_D
