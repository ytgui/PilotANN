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


def graph_sampling(indptr: torch.Tensor,
                   indices: torch.Tensor,
                   n_samples: int,
                   n_hops: int):
    return extension.sampling_cpu(
        indptr, indices, n_samples, n_hops
    )


def traverse_cpu(output_I: torch.Tensor,
                 output_D: torch.Tensor,
                 initial_I: torch.Tensor,
                 initial_D: torch.Tensor,
                 indptr: torch.Tensor,
                 indices: torch.Tensor,
                 storage: torch.Tensor,
                 query: torch.Tensor,
                 ef_search: int):
    extension.traverse_cpu(
        output_I, output_D, initial_I, initial_D,
        indptr, indices, storage, query, ef_search
    )


def traverse_cuda(output_I: torch.Tensor,
                  output_D: torch.Tensor,
                  indptr: torch.Tensor,
                  indices: torch.Tensor,
                  mapping: torch.Tensor,
                  storage: torch.Tensor,
                  query: torch.Tensor,
                  initial_I: torch.Tensor,
                  initial_D: torch.Tensor,
                  n_neighbors: int):
    extension.traverse_cuda(
        output_I, output_D, indptr, indices, mapping,
        storage, query, initial_I, initial_D, n_neighbors
    )


def traverse_refine(output_I: torch.Tensor,
                    output_D: torch.Tensor,
                    buffer_I: torch.Tensor,
                    buffer_D: torch.Tensor,
                    initial_I: torch.Tensor,
                    initial_D: torch.Tensor,
                    subgraph: list[torch.Tensor],
                    fullgraph: list[torch.Tensor],
                    storage: torch.Tensor,
                    query: torch.Tensor,
                    ef_search: int,
                    d_principle: int):
    extension.traverse_refine(
        output_I, output_D, buffer_I, buffer_D,
        initial_I, initial_D, subgraph, fullgraph,
        storage, query, ef_search, d_principle
    )
