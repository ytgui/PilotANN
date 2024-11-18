import torch
from pilot_ann import utils, proto


def evaluate_proto(method: str):
    k = 10
    ef_search = 32
    batch_size = 256

    # init
    loader = utils.DataLoader('fuzz-16k')
    query, target = loader.load_query(
        n_queries=batch_size, k=k
    )
    storage = loader.load_storage()

    # index
    if method == 'nsw':
        index = proto.IndexNSW(d_model=loader.d_model)
    elif method == 'svd':
        d_principle = 8 * (loader.d_model // 16)
        index = proto.IndexSVD(
            d_model=loader.d_model, d_major=d_principle
        )
    elif method == 'sampled':
        index = proto.IndexSampled(
            d_model=loader.d_model, sample_ratio=1/2
        )
    elif method == 'staged':
        d_principle = 8 * (loader.d_model // 16)
        index = proto.IndexStaged(
            d_model=loader.d_model, d_major=d_principle,
            sample_ratio=1/2
        )
    else:
        raise NotImplementedError
    index.train(storage)

    # search
    output = index.search(
        query, k=k, ef_search=ef_search
    )
    score = utils.recall(
        output['topk'], target=target
    )
    print('[{} d={}] score@{}: {:.3f}'
          .format(method, loader.d_model, k, score))
    assert score >= 0.1


def test_proto_1():
    for method in [
        'nsw', 'svd', 'sampled', 'staged'
    ]:
        evaluate_proto(method=method)

    #
    print('[PASS] test_proto_1()')


def main():
    test_proto_1()


if __name__ == '__main__':
    main()
