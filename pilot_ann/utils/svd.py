import torch


def svd(x: torch.Tensor, max_size: int):
    assert x.dim() == 2

    #
    if x.size(0) <= max_size:
        U, S, V = torch.svd(x)
        x = torch.matmul(U, torch.diag(S))
    else:
        indices = torch.randint(
            high=x.size(0), size=[max_size]
        )
        _, _, V = torch.svd(
            torch.index_select(x, dim=0, index=indices)
        )
        x = torch.matmul(x, V)

    #
    return x, V.contiguous()
