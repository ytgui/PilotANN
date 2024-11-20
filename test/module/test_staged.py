import torch
from pilot_ann import utils, layers


def evaluate_stage(entry_method: str,
                   graph_method: str):
    k = 10
    ef_search = 32
    batch_size = 1024
    cuda_device = 'cuda'
    sample_ratio = 1/4

    # init
    loader = utils.DataLoader('deep-64k')
    query, target = loader.load_query(
        n_queries=batch_size, k=k
    )
    storage = loader.load_storage()
    d_principle = 8 * (loader.d_model // 16)

    # index
    index = layers.IndexStaged(
        d_model=loader.d_model,
        d_principle=d_principle,
        sample_ratio=sample_ratio,
        entry_method=entry_method,
        graph_method=graph_method
    )
    index.train(x=storage)
    index.to(cuda_device)

    # search
    I, D = index.search(
        query, k=k, ef_search=ef_search
    )
    score = utils.recall(I.cpu(), target=target)
    print('{}: d_model={}, recall@{}: {:.3f}'.format(
        graph_method, loader.d_model, k, score
    ))

    # check recall
    assert score > 0.90

    # check distance
    target = torch.cdist(
        query.unsqueeze(1), storage[I.cpu()], p=2.0
    )
    assert torch.allclose(
        D, target.squeeze(1).square(), atol=1e-3
    )


def main():
    evaluate_stage(
        entry_method='random32', graph_method='nsw32'
    )

    #
    print('[PASS] test_staged_1()')


if __name__ == '__main__':
    main()
