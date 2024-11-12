import faiss
import torch
from pilot_ann import kernels


def graph_nsg_init(x: torch.Tensor,
                   n_neighbors: int,
                   ef_construct: int):
    if x.dim() != 2:
        raise RuntimeError
    n_nodes = x.size(0)
    d_model = x.size(-1)

    # index
    index = faiss.IndexNSGFlat(
        d_model, n_neighbors
    )
    index.GK = ef_construct
    index.add(x.detach().cpu().numpy())
    graph = index.nsg.get_final_graph()

    # graph
    indptr = [
        i * n_neighbors for i in range(n_nodes + 1)
    ]
    indices = [
        -1 for _ in range(n_nodes * n_neighbors)
    ]
    for u in range(n_nodes):
        for i in range(n_neighbors):
            v = graph.at(u, i)
            if v < 0:
                break
            indices[u * n_neighbors + i] = v
    assert indptr[-1] == len(indices)
    assert len(indptr) == n_nodes + 1
    assert len(indices) % n_neighbors == 0

    #
    return indptr, indices


def graph_nsw_init(x: torch.Tensor,
                   n_neighbors: int,
                   ef_construct: int):
    if x.dim() != 2:
        raise RuntimeError
    n_nodes = x.size(0)
    d_model = x.size(-1)

    # index
    index = faiss.IndexHNSWFlat(
        d_model, n_neighbors // 2
    )
    index.hnsw.efConstruction = ef_construct
    index.add(x.detach().cpu().numpy())

    # graph
    indptr = [
        i * n_neighbors for i in range(n_nodes + 1)
    ]
    indices = [
        -1 for _ in range(n_nodes * n_neighbors)
    ]
    for u in range(n_nodes):
        left = index.hnsw.offsets.at(u)
        for i in range(n_neighbors):
            v = index.hnsw.neighbors.at(left + i)
            if v < 0:
                break
            indices[u * n_neighbors + i] = v
    assert indptr[-1] == len(indices)
    assert len(indptr) == n_nodes + 1
    assert len(indices) % n_neighbors == 0

    #
    return indptr, indices


def graph_init(x: torch.Tensor,
               graph_type: str,
               n_neighbors: int,
               ef_construct: int = 80):
    if graph_type == 'nsg':
        return graph_nsg_init(
            x, n_neighbors, ef_construct=ef_construct
        )
    elif graph_type == 'nsw':
        return graph_nsw_init(
            x, n_neighbors, ef_construct=ef_construct
        )
    else:
        raise RuntimeError


def subgraph_init(indptr: list,
                  indices: list,
                  storage: torch.Tensor,
                  graph_type: str,
                  n_samples: int,
                  n_neighbors: int,
                  n_hops: int = 1):
    # sampling
    sampled = kernels.graph_sampling(
        indptr=indptr, indices=indices,
        n_samples=n_samples, n_hops=n_hops
    )
    nodelist, mapping = sampled

    # subgraph
    subgraph = graph_init(
        x=storage[nodelist],
        graph_type=graph_type,
        n_neighbors=n_neighbors
    )
    sg_indptr, sg_indices = subgraph

    # remapping
    new_indptr, new_indices = [0], []
    for u in range(len(indptr) - 1):
        if mapping[u] != -1:
            u = mapping[u]
            for v in sg_indices[
                sg_indptr[u]:sg_indptr[u + 1]
            ]:
                new_indices.append(nodelist[v])
        new_indptr.append(len(new_indices))
    assert new_indptr[-1] == len(new_indices)
    assert len(new_indptr) == len(indptr)

    #
    return new_indptr, new_indices, nodelist, mapping
