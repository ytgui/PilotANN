import torch
from torch import nn
from pilot_ann import proto


class IndexStaged(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_major: int,
                 sample_ratio: float,
                 method: str = 'nsw32'):
        nn.Module.__init__(self)
        #
        self.d_model = d_model
        self.d_major = d_major
        self.sample_ratio = sample_ratio
        # svd
        self.VT: torch.Tensor
        self.stage_1 = proto.IndexSampled(
            d_model=d_major,
            sample_ratio=sample_ratio,
            method=method
        )
        self.stage_2 = proto.IndexSampled(
            d_model=d_model,
            sample_ratio=sample_ratio,
            method=method
        )
        self.stage_3 = proto.IndexNSW(
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
        self.stage_3.train(x)
        self.stage_2.train(x)
        self.stage_1.train(
            x=x[:, :self.d_major],
            indptr=self.stage_2.indptr,
            indices=self.stage_2.indices
        )

    def search(self,
               query: torch.FloatTensor,
               k: int, ef_search: int):
        assert query.dim() == 2

        # transform
        query = torch.matmul(query, self.VT.T)

        # traverse 1
        output_1 = self.stage_1.search(
            query=query[:, :self.d_major],
            k=4 * ef_search, ef_search=4 * ef_search
        )

        # traverse 2
        def entry(query: torch.Tensor,
                  ef_search: int):
            return output_1['topk'].tolist()
        self.stage_2.entry = entry
        output_2 = self.stage_2.search(
            query=query, k=k, ef_search=ef_search
        )

        # traverse 3
        def entry(query: torch.Tensor,
                  ef_search: int):
            return output_2['topk'].tolist()

        def closedlist(query: torch.Tensor):
            visited = output_2['visited']
            return [
                list(item) for item in visited
            ]
        self.stage_3.entry = entry
        self.stage_3.closedlist = closedlist
        output_3 = self.stage_3.search(
            query=query, k=k, ef_search=ef_search
        )

        # output
        output = {
            'topk': output_3['topk'],
            'steps-1': output_1['n_steps'],
            'steps-2': output_2['n_steps'],
            'steps-3': output_3['n_steps'],
            'visited-1': output_1['n_visited'],
            'visited-2': output_2['n_visited'],
            'visited-3': output_3['n_visited'],
            'computed-1': output_1['n_computed'],
            'computed-2': output_2['n_computed'],
            'computed-3': output_3['n_computed'],
        }
        return output
