import torch
from pilot_ann import utils


def evaluate(graph_type: str):
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
        128: 0.75, 512: 0.35, 1024: 0.25
    }
    assert score >= next(
        v for k, v in thresholds.items() if loader.d_model <= k
    )


def test_graph_init():
    for graph_type in ['nsg', 'nsw']:
        evaluate(graph_type=graph_type)

    #
    print('[PASS] test_graph_init()')


def main():
    test_graph_init()


if __name__ == '__main__':
    main()
