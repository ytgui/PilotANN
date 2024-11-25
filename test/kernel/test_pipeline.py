import torch
from pilot_ann import utils, kernels


def evaluate_pipeline(graph_type: str,
                      cuda_device: str):
    k = 10
    ef_search = 64
    n_neighbors = 32
    batch_size = 1024
    chunk_size = 128

    # init
    loader = utils.DataLoader('fuzz-64k')
    query, target = loader.load_query(
        n_queries=batch_size, k=k
    )
    storage = loader.load_storage()
    d_principle = loader.d_model

    # graph
    graph = utils.graph_init(
        storage, graph_type=graph_type,
        n_neighbors=n_neighbors
    )
    subgraph = utils.subgraph_init(
        indptr=graph[0], indices=graph[1],
        storage=storage, graph_type=graph_type,
        n_samples=loader.n_storage // 4,
        n_neighbors=n_neighbors
    )
    nodelist, mapping = subgraph[2:]

    # parameters
    graph = [
        torch.LongTensor(graph[0]),
        torch.LongTensor(graph[1])
    ]
    subgraph_cuda = [
        torch.LongTensor(subgraph[0]).to(cuda_device),
        torch.LongTensor(subgraph[1]).to(cuda_device),
        torch.LongTensor(mapping).to(cuda_device)
    ]
    subgraph = [
        torch.LongTensor(subgraph[0]),
        torch.LongTensor(subgraph[1])
    ]
    query_cuda = query.to(cuda_device)
    storage_cuda = storage[nodelist].to(cuda_device)

    # traverse
    initial_I = torch.randint(
        high=loader.n_storage,
        size=[batch_size, ef_search],
        device=cuda_device
    )
    distances = torch.cdist(
        query_cuda, storage.to(cuda_device), p=2.0
    )
    initial_D = torch.gather(
        distances.square(), dim=-1, index=initial_I
    )
    topk_I, topk_D = kernels.pipeline(
        graph=graph,
        subgraph=subgraph,
        subgraph_cuda=subgraph_cuda,
        storage=storage, storage_cuda=storage_cuda,
        query=query, query_cuda=query_cuda,
        initial_I=initial_I, initial_D=initial_D,
        n_neighbors=n_neighbors, k=k, ef_search=ef_search,
        d_principle=d_principle, chunk_size=chunk_size
    )
    score = utils.recall(topk_I, target=target)
    print('{}: d_model={}, recall@{}: {:.3f}'.format(
        graph_type, loader.d_model, k, score
    ))

    # check recall
    thresholds = {
        128: 0.50, 512: 0.30, 1024: 0.20
    }
    assert score >= next(
        v for k, v in thresholds.items() if loader.d_model <= k
    )

    # check distances
    target = torch.cdist(
        query.unsqueeze(1), storage[topk_I], p=2.0
    )
    assert torch.allclose(
        topk_D, target.squeeze(1).square(), atol=1e-3
    )


def test_pipeline():
    evaluate_pipeline(
        graph_type='nsw', cuda_device='cuda'
    )

    #
    print('[PASS] test_pipeline()')


def main():
    test_pipeline()


if __name__ == '__main__':
    main()
