import torch
import random
from pilot_ann import utils, layers


def evaluate(method: str, device: str):
    batch_size = random.randint(256, 1024)

    # init
    loader = utils.DataLoader('fuzz-64k')
    query, target = loader.load_query(
        n_queries=batch_size, k=100
    )
    storage = loader.load_storage()

    # entry
    index = layers.IndexEntry(
        d_model=loader.d_model, method=method
    )
    index = index.train(storage)
    index = index.to(device)

    # search
    EP_I, EP_D = index.search(
        query=query.to(device)
    )
    EP_I, EP_D = EP_I.cpu(), EP_D.cpu()

    # check I
    assert EP_I.size() == EP_D.size()
    assert EP_I.size(0) == batch_size
    score = utils.recall(
        EP_I, target=target[:, :EP_I.size(-1)]
    )
    assert score != 0.0

    # check D
    target = torch.cdist(
        query.unsqueeze(1), storage[EP_I], p=2.0
    )
    target = target.squeeze(1).square()
    assert torch.allclose(EP_D, target, atol=1e-3)


def test_entry_1():
    for method in ['random32', 'random64']:
        evaluate(method=method, device='cpu')

    #
    print('[PASS] test_entry_1()')


def test_entry_2():
    for method in [
        'random32', 'router16x32', 'router32x32'
    ]:
        evaluate(method=method, device='cuda')

    #
    print('[PASS] test_entry_2()')


def main():
    test_entry_1()
    test_entry_2()


if __name__ == '__main__':
    main()
