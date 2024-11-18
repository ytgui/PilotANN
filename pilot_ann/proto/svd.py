import torch
from torch import nn
from pilot_ann import proto


class IndexSVD(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_major: int,
                 method: str = 'nsw32'):
        nn.Module.__init__(self)
        #
        self.d_model = d_model
        self.d_major = d_major
        #
        self.VT: torch.Tensor
        self.major = proto.IndexNSW(
            d_model=d_major, method=method
        )
        self.finer = proto.IndexNSW(
            d_model=d_model, method=method
        )

    def train(self, x: torch.Tensor):
        assert x.dim() == 2

        # decompose
        U, S, V = torch.svd(x)
        self.VT = V.T.contiguous()

        # transform
        x = torch.matmul(U, torch.diag(S))

        # train
        self.finer.train(x)
        self.major.train(
            x=x[:, :self.d_major],
            indptr=self.finer.indptr,
            indices=self.finer.indices
        )

    def search(self,
               query: torch.FloatTensor,
               k: int, ef_search: int):
        assert query.dim() == 2

        # transform
        query = torch.matmul(query, self.VT.T)

        # traverse 1
        output_1 = self.major.search(
            query=query[:, :self.d_major],
            k=4 * ef_search, ef_search=4 * ef_search
        )

        # new entry
        def entry(query: torch.Tensor,
                  ef_search: int):
            return output_1['topk'].tolist()

        # traverse 2
        self.finer.entry = entry
        output_2 = self.finer.search(
            query=query, k=k, ef_search=ef_search
        )

        # output
        output = {
            'topk': output_2['topk'],
            'major-steps': output_1['n_steps'],
            'finer-steps': output_2['n_steps'],
            'major-visited': output_1['n_visited'],
            'finer-visited': output_2['n_visited']
        }
        return output
