import torch
from pilot_ann import utils, kernels


def evaluate_cpu(graph_type: str, sampling: bool):
    k = 10
    ef_search = 128
    n_neighbors = 64
    batch_size = 1024

    # init
    loader = utils.DataLoader('fuzz-64k')
    storage = loader.load_storage()
    query, target = loader.load_query(
        n_queries=batch_size, k=k
    )

    # graph
    indptr, indices = utils.graph_init(
        storage, graph_type, n_neighbors=n_neighbors
    )
    mapping = list(range(loader.n_storage))

    # sampling
    if sampling:
        subgraph = utils.subgraph_init(
            indptr=indptr, indices=indices,
            storage=storage, graph_type=graph_type,
            n_samples=loader.n_storage // 4,
            n_neighbors=n_neighbors
        )
        indptr, indices, nodelist, mapping = subgraph
        assert max(mapping) + 1 == len(nodelist)
        assert len(set(mapping)) == len(nodelist) + 1
        assert all(
            mapping[x] != -1 for x in nodelist
        )

    # traverse
    indptr = torch.LongTensor(indptr)
    indices = torch.LongTensor(indices)
    distances = torch.square(
        torch.cdist(query, storage, p=2.0)
    )
    output_I = torch.empty(
        size=[batch_size, k], dtype=torch.long
    )
    output_D = torch.empty(
        size=[batch_size, k], dtype=torch.float
    )
    initial_I = torch.randint(
        high=loader.n_storage,
        size=[batch_size, 4 * ef_search], dtype=torch.long
    )
    initial_D = torch.gather(
        distances, dim=-1, index=initial_I.to(torch.long)
    )
    kernels.traverse_cpu(
        output_I=output_I, output_D=output_D,
        indptr=indptr, indices=indices, storage=storage,
        query=query, initial_I=initial_I, initial_D=initial_D,
        n_neighbors=n_neighbors, ef_search=ef_search
    )
    score = utils.recall(output_I, target=target)
    print('{}: d_model={}, recall@{}: {:.3f}'.format(
        graph_type, loader.d_model, k, score
    ))

    # check recall
    thresholds = {
        128: 0.60, 512: 0.35, 1024: 0.20
    }
    if sampling:
        thresholds = {
            k: v / 4.0 for k, v in thresholds.items()
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


def test_traverse_cpu_1():
    for graph_type in ['nsg', 'nsw']:
        evaluate_cpu(
            graph_type=graph_type, sampling=False
        )

    #
    print('[PASS] test_traverse_cpu_1()')


def test_traverse_cpu_2():
    for graph_type in ['nsg', 'nsw']:
        evaluate_cpu(
            graph_type=graph_type, sampling=True
        )

    #
    print('[PASS] test_traverse_cpu_2()')


def main():
    test_traverse_cpu_1()
    test_traverse_cpu_2()


if __name__ == '__main__':
    main()
