import copy
import heapq
import torch
import random
from pilot_ann import kernels


def test_heapq():
    heap_size = random.randint(1, 1024)

    # check pop
    heap_keys = torch.zeros(
        [heap_size], dtype=torch.float
    )
    heap_values = torch.zeros(
        [heap_size], dtype=torch.long
    )
    for _ in range(heap_size):
        kernels.heapq_pop(
            heap_keys, heap_values
        )
    assert torch.allclose(
        heap_keys, torch.full_like(
            heap_keys, fill_value=float('inf')
        )
    )
    assert torch.allclose(
        heap_values, torch.full_like(
            heap_values, fill_value=-1
        )
    )

    # check pushpop
    heap_keys = torch.zeros(
        [heap_size], dtype=torch.float
    )
    heap_values = torch.zeros(
        [heap_size], dtype=torch.long
    )
    for _ in range(heap_size):
        kernels.heapq_pushpop(
            heap_keys, heap_values, key=-1.0, value=-1
        )
    assert torch.allclose(
        heap_keys, torch.zeros_like(heap_keys)
    )
    assert torch.allclose(
        heap_values, torch.zeros_like(heap_values)
    )
    for _ in range(heap_size):
        kernels.heapq_pushpop(
            heap_keys, heap_values, key=1.0, value=-1
        )
    assert torch.allclose(
        heap_keys, torch.full_like(
            heap_keys, fill_value=1.0
        )
    )
    assert torch.allclose(
        heap_values, torch.full_like(
            heap_values, fill_value=-1
        )
    )

    # check heapify
    def check_heapify(heap: list):
        old = copy.deepcopy(heap)
        heapq.heapify(heap)
        if old != heap:
            return False
        return True

    # check replace
    heap_keys = torch.zeros(
        [heap_size], dtype=torch.float
    )
    heap_values = torch.zeros(
        [heap_size], dtype=torch.long
    )
    for _ in range(heap_size):
        kernels.heapq_replace(
            heap_keys, heap_values, key=-1.0, value=-1
        )
    assert check_heapify(heap_keys.tolist())
    assert heap_keys[0].item() == -1

    #
    print('[PASS] test_heapq()')


def test_bitmask():
    max_key = 16384
    batch_size = random.randint(1, 1024)

    # init
    keys = torch.randint(
        high=max_key, size=[batch_size], dtype=torch.long
    )
    tensor = torch.flatten(
        kernels.bitmask_put(keys, n=max_key)
    )

    # check count
    n_unique = keys.unique().size(0)
    n_nonzeros = torch.count_nonzero(tensor).item()
    assert n_unique == n_nonzeros

    # check element
    for x in keys.tolist():
        assert tensor[x].item()

    #
    print('[PASS] test_bitmask()')


def test_distance():
    d_model = 16 * random.randint(1, 16)
    n_storage = random.randint(16384, 65536)
    batch_size = random.randint(1, 1024)

    # init
    query = torch.randn([d_model])
    storage = torch.randn([n_storage, d_model])
    nodelist = torch.randint(
        high=n_storage, size=[batch_size], dtype=torch.long
    )

    # check
    target = torch.cdist(
        query.unsqueeze(dim=0), storage[nodelist], p=2.0
    )
    output = kernels.square_dist(query, storage, nodelist)
    assert torch.allclose(target.square(), output, atol=1e-3)

    #
    print('[PASS] test_distance()')


def main():
    test_heapq()
    test_bitmask()
    test_distance()


if __name__ == '__main__':
    main()
