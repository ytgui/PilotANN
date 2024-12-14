import torch
from torch import nn
from pilot_ann import utils, kernels, layers


class GraphModule(nn.Module):
    def __init__(self,
                 indptr: list,
                 indices: list):
        nn.Module.__init__(self)
        #
        self.register_buffer(
            'indptr', torch.LongTensor(indptr)
        )
        self.register_buffer(
            'indices', torch.LongTensor(indices)
        )


class IndexStaged(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_principle: int,
                 sample_ratio: int,
                 graph_method: str = 'nsw32',
                 entry_method: str = 'random32'):
        nn.Module.__init__(self)
        assert d_principle <= d_model
        # params
        self.d_model = d_model
        self.d_principle = d_principle
        self.sample_ratio = sample_ratio
        self.graph_method = graph_method
        self.entry_method = entry_method
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
        self.subgraph: GraphModule
        self.fullgraph: GraphModule
        self.subgraph_cuda: GraphModule
        # entry
        self.entry = layers.IndexEntry(
            d_model=d_principle, method=entry_method
        )

    @classmethod
    def load(self, ckpt_file: any):
        checkpoint = torch.load(ckpt_file)
        state_dict = checkpoint['state_dict']

        # index
        config = checkpoint['config']
        index = IndexStaged(
            d_model=config['d_model'],
            d_principle=config['d_principle'],
            graph_method=config['graph_method'],
            entry_method=config['entry_method'],
            sample_ratio=-1.0
        )

        # init buffers
        index.register_buffer(
            'VT', state_dict['VT']
        )
        index.register_buffer(
            'storage', state_dict['storage']
        )
        index.register_buffer(
            'mapping', state_dict['mapping']
        )
        index.register_buffer(
            'nodelist', state_dict['nodelist']
        )
        index.register_module(
            'subgraph', GraphModule(
                indptr=state_dict['subgraph.indptr'],
                indices=state_dict['subgraph.indices']
            )
        )
        index.register_module(
            'fullgraph', GraphModule(
                indptr=state_dict['fullgraph.indptr'],
                indices=state_dict['fullgraph.indices']
            )
        )
        if 'entry.entry_nodes' in state_dict:
            index.entry.register_buffer(
                'entry_nodes', state_dict['entry.entry_nodes']
            )
        if 'entry.entry_vectors' in state_dict:
            index.entry.register_buffer(
                'entry_vectors', state_dict['entry.entry_vectors']
            )
        if 'entry.route_vectors' in state_dict:
            index.entry.register_buffer(
                'route_vectors', state_dict['entry.route_vectors']
            )
        return index

    def dump(self):
        return {
            'config': {
                'd_model': self.d_model,
                'd_principle': self.d_principle,
                'graph_method': self.graph_method,
                'entry_method': self.entry_method
            },
            'state_dict': self.state_dict()
        }

    def to(self, device: str):
        self.cuda_device = device

        # copy
        self.entry.to(device=device)
        self.mapping = self.mapping.to(device=device)
        self.subgraph_cuda = GraphModule(
            indptr=self.subgraph.indptr,
            indices=self.subgraph.indices
        )
        self.subgraph_cuda.to(device=device)
        storage = torch.index_select(
            self.storage, dim=0, index=self.nodelist
        )
        storage = storage[:, :self.d_principle]
        self.storage_cuda = storage.to(device=device)

    def train(self, x: torch.Tensor):
        assert x.dim() == 2
        assert x.size(-1) == self.d_model
        assert 0.0 < self.sample_ratio <= 1.0

        # decompose
        print('svd', self.d_principle)
        x, V = utils.svd(x, max_size=10_000_000)
        self.register_buffer('storage', x)
        self.register_buffer(
            'VT', V.T.contiguous()
        )

        # fullgraph
        print('build', self.graph_type)
        graph = utils.graph_init(
            x, graph_type=self.graph_type,
            n_neighbors=self.n_neighbors
        )
        self.register_module(
            'fullgraph', GraphModule(
                indptr=graph[0], indices=graph[1]
            )
        )

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
        self.register_module(
            'subgraph', GraphModule(
                indptr=subgraph[0], indices=subgraph[1]
            )
        )
        self.register_buffer(
            'nodelist', torch.LongTensor(subgraph[2])
        )
        self.register_buffer(
            'mapping', torch.LongTensor(subgraph[-1])
        )

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
            graph=[
                self.fullgraph.indptr,
                self.fullgraph.indices
            ],
            storage=self.storage,
            subgraph=[
                self.subgraph.indptr,
                self.subgraph.indices
            ],
            subgraph_cuda=[
                self.subgraph_cuda.indptr,
                self.subgraph_cuda.indices,
                self.mapping
            ],
            storage_cuda=self.storage_cuda,
            query=query, query_cuda=query_cuda,
            initial_I=EP_I, initial_D=EP_D, k=k,
            n_neighbors=self.n_neighbors, ef_search=ef_search,
            d_principle=self.d_principle, chunk_size=chunk_size
        )
        return output_I, output_D
