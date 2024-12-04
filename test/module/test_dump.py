import torch
from pilot_ann import utils, layers


def dump(loader: utils.DataLoader, output: str):
    d_principle = 64
    sample_ratio = 0.25

    # index
    index = layers.IndexStaged(
        d_model=loader.d_model,
        d_principle=d_principle,
        sample_ratio=sample_ratio
    )
    index.train(x=loader.storage)

    # dump
    torch.save(index.dump(), f=output)


def test_dump_1():
    k = 10
    ef_search = 32
    batch_size = 1024
    cuda_device = 'cuda'
    output = 'tmp.bin'

    # init
    loader = utils.DataLoader('deep-64k')
    query, target = loader.load_query(
        n_queries=batch_size, k=k
    )

    # dump
    dump(loader=loader, output=output)
    index = layers.IndexStaged.load(ckpt_file=output)

    # search
    index.to(device=cuda_device)
    I, D = index.search(
        query, k=k, ef_search=ef_search
    )
    score = utils.recall(I.cpu(), target=target)
    print('{}: d_model={}, recall@{}: {:.3f}'.format(
        index.graph_method, loader.d_model, k, score
    ))
    assert score > 0.90

    #
    print('[PASS] test_dump_1()')


def main():
    test_dump_1()


if __name__ == '__main__':
    main()
