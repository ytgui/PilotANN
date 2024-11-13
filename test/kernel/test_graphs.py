import torch
from pilot_ann import utils


def evaluate(graph_type: str, sampling: bool):
    k = 10
    ef_search = 32
    n_neighbors = 64
    batch_size = 256

    # init
    loader = utils.DataLoader('fuzz-16k')
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
        storage = storage[nodelist]

    # search
    output = torch.LongTensor([
        utils.search_simple(
            indptr=indptr, indices=indices,
            mapping=mapping, storage=storage,
            query=query[i], k=k, ef_search=ef_search
        )['topk']
        for i in range(batch_size)
    ])
    score = utils.recall(output, target=target)
    print('{}: d_model={}, recall@{}: {:.3f}'.format(
        graph_type, loader.d_model, k, score
    ))

    # check
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


def test_graph_init():
    for graph_type in ['nsg', 'nsw']:
        evaluate(
            graph_type=graph_type, sampling=False
        )

    #
    print('[PASS] test_graph_init()')


def test_subgraph_init():
    for graph_type in ['nsg', 'nsw']:
        evaluate(
            graph_type=graph_type, sampling=True
        )

    #
    print('[PASS] test_subgraph_init()')


def main():
    test_graph_init()
    test_subgraph_init()


if __name__ == '__main__':
    main()
