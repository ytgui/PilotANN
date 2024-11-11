import torch
from pilot_ann import extension


def heapq_pop(keys: torch.Tensor,
              values: torch.Tensor):
    return extension.heapq_pop_cpu(keys, values)


def heapq_pushpop(keys: torch.Tensor,
                  values: torch.Tensor,
                  key: float, value: int):
    return extension.heapq_pushpop_cpu(
        keys, values, key, value
    )


def heapq_replace(keys: torch.Tensor,
                  values: torch.Tensor,
                  key: float, value: int):
    return extension.heapq_replace_cpu(
        keys, values, key, value
    )


def bitmask_put(keys: torch.Tensor, n: int):
    return extension.bitmask_put_cpu(keys, n)


def square_dist(query: torch.Tensor,
                storage: torch.Tensor,
                nodelist: torch.Tensor):
    return extension.square_dist_cpu(
        query, storage, nodelist
    )
