import torch
from pilot_ann import utils, kernels


def evaluate_refine(graph_type: str):
    k = 10
    ef_search = 32
    n_neighbors = 64
    batch_size = 1024

    # init
    loader = utils.DataLoader('fuzz-64k')
    storage = loader.load_storage()
    query, target = loader.load_query(
        n_queries=batch_size, k=k
    )
    d_principle = 8 * (loader.d_model // 16)

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

    # traverse
    graph = [
        torch.LongTensor(graph[0]),
        torch.LongTensor(graph[1])
    ]
    subgraph = [
        torch.LongTensor(subgraph[0]),
        torch.LongTensor(subgraph[1])
    ]
    output_I = torch.empty(
        size=[batch_size, k], dtype=torch.long
    )
    output_D = torch.empty(
        size=[batch_size, k], dtype=torch.float
    )
    buffer_I = torch.empty(
        size=[batch_size, ef_search], dtype=torch.long
    )
    buffer_D = torch.empty(
        size=[batch_size, ef_search], dtype=torch.float
    )
    initial_I = torch.randint(
        high=loader.n_storage, size=[batch_size, ef_search]
    )
    distances = torch.cdist(
        query[:, :d_principle], storage[:, :d_principle], p=2.0
    )
    initial_D = torch.gather(
        distances.square(), dim=-1, index=initial_I
    )
    kernels.traverse_refine(
        output_I=output_I, output_D=output_D,
        buffer_I=buffer_I, buffer_D=buffer_D,
        initial_I=initial_I, initial_D=initial_D,
        subgraph=subgraph, fullgraph=graph, storage=storage,
        query=query, d_principle=d_principle, ef_search=ef_search
    )
    score = utils.recall(output_I, target=target)
    print('{}: d_model={}, recall@{}: {:.3f}'.format(
        graph_type, loader.d_model, k, score
    ))

    # check recall
    thresholds = {
        128: 0.60, 512: 0.35, 1024: 0.20
    }
    assert score >= next(
        v for k, v in thresholds.items() if loader.d_model <= k
    )

    # check distance
    target = torch.cdist(
        query.unsqueeze(1), storage[output_I], p=2.0
    )
    assert torch.allclose(
        output_D, target.squeeze(1).square(), atol=1e-3
    )

    # check residual
    target = torch.cdist(
        query.unsqueeze(1), storage[buffer_I], p=2.0
    )
    assert torch.allclose(
        buffer_D, target.squeeze(1).square(), atol=1e-3
    )


def test_traverse_refine():
    for graph_type in ['nsg', 'nsw']:
        evaluate_refine(graph_type=graph_type)

    #
    print('[PASS] test_traverse_refine()')


def main():
    test_traverse_refine()


if __name__ == '__main__':
    main()
