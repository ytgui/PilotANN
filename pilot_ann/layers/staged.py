import torch
from torch import nn
from pilot_ann import utils, kernels, layers


class IndexStaged(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_principle: int,
                 sample_ratio: int,
                 graph_method: str = 'nsw32',
                 entry_method: str = 'random32'):
        nn.Module.__init__(self)
        # params
        self.d_model = d_model
        self.d_principle = d_principle
        self.sample_ratio = sample_ratio
        self.cuda_device = None
        # builder
        for graph_type in ['nsw', 'nsg']:
            if graph_method.startswith(graph_type):
                self.graph_type = graph_type
                self.n_neighbors = int(
                    graph_method.removeprefix(graph_type)
                )
                break
        # storage
        self.VT: torch.Tensor
        self.storage: torch.Tensor
        self.mapping: torch.Tensor
        self.nodelist: torch.Tensor
        self.storage_cuda: torch.Tensor
        # graphs
        self.subgraph: list[torch.Tensor]
        self.fullgraph: list[torch.Tensor]
        self.subgraph_cuda: list[torch.Tensor]
        # entry
        self.entry = layers.IndexEntry(
            d_model=d_principle, method=entry_method
        )

    def to(self, device: str):
        self.cuda_device = device

        # copy
        self.entry.to(device=device)
        assert len(self.subgraph) == 2
        self.subgraph_cuda = [
            self.subgraph[0].to(device=device),
            self.subgraph[1].to(device=device),
            self.mapping.to(device=device)
        ]
        storage = torch.index_select(
            self.storage, dim=0, index=self.nodelist
        )
        storage = storage[:, :self.d_principle]
        self.storage_cuda = storage.to(device=device)

    def train(self, x: torch.Tensor):
        assert x.dim() == 2
        assert x.size(-1) == self.d_model

        # decompose
        U, S, V = torch.svd(x)
        self.VT = V.T.contiguous()

        # transform
        print('svd', self.d_principle)
        x = torch.matmul(U, torch.diag(S))

        # fullgraph
        print('build', self.graph_type)
        graph = utils.graph_init(
            x, graph_type=self.graph_type,
            n_neighbors=self.n_neighbors
        )
        self.fullgraph = [
            torch.LongTensor(graph[0]),
            torch.LongTensor(graph[1])
        ]
        self.storage = x

        # sampling
        print('sampling', self.sample_ratio)
        subgraph = utils.subgraph_init(
            indptr=graph[0], indices=graph[1],
            storage=x, graph_type=self.graph_type,
            n_samples=round(
                self.sample_ratio * x.size(0)
            ),
            n_neighbors=self.n_neighbors
        )
        self.subgraph = [
            torch.LongTensor(subgraph[0]),
            torch.LongTensor(subgraph[1])
        ]
        self.nodelist = torch.LongTensor(subgraph[2])
        self.mapping = torch.LongTensor(subgraph[-1])

        # entry
        self.entry.train(x[:, :self.d_principle])

    def search(self,
               query: torch.FloatTensor,
               k: int, ef_search: int,
               chunk_size: int = 128):
        assert query.dim() == 2

        # svd
        query = torch.matmul(query, self.VT.T)

        # entry
        query_cuda = query[:, :self.d_principle].to(
            device=self.cuda_device
        )
        EP_I, EP_D = self.entry.search(query_cuda)

        # traverse
        output_I, output_D = kernels.pipeline(
            graph=self.fullgraph,
            storage=self.storage,
            subgraph=self.subgraph,
            storage_cuda=self.storage_cuda,
            subgraph_cuda=self.subgraph_cuda,
            query=query, query_cuda=query_cuda,
            initial_I=EP_I, initial_D=EP_D, k=k,
            n_neighbors=self.n_neighbors, ef_search=ef_search,
            d_principle=self.d_principle, chunk_size=chunk_size
        )
        return output_I, output_D
