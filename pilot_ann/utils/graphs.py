import faiss
import torch


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
