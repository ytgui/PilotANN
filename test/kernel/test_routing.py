import torch
import random
from pilot_ann import utils, kernels


def test_routing_1():
    n_routers = random.choice([8, 16])
    n_leaves = random.choice([32, 64])
    batch_size = random.randrange(1, 1024)
    cuda_device = 'cuda'

    # init
    loader = utils.DataLoader('fuzz-16k')
    storage = loader.load_storage()
    query = loader.load_query(batch_size, k=10)[0]
    init_nodes = random.choices(
        range(loader.n_storage), k=n_leaves
    )

    # router
    router, entry_nodes = utils.router_init(
        storage, init_nodes=init_nodes.copy(),
        n_routers=n_routers, n_leaves=n_leaves
    )
    assert set(init_nodes).intersection(
        entry_nodes.flatten().tolist()
    )
    assert entry_nodes.size() == (n_routers, n_leaves)
    assert router.size() == (n_routers, loader.d_model)

    # kernel
    route, distances = kernels.routing_cuda(
        query=query.to(cuda_device),
        route_vectors=router.to(cuda_device),
        entry_nodes=entry_nodes.to(cuda_device),
        entry_vectors=storage[entry_nodes].to(cuda_device)
    )

    # target
    indices = torch.argmin(
        torch.cdist(query, router, p=2.0), dim=-1
    )
    entries = torch.index_select(
        entry_nodes, dim=0, index=indices
    )
    target = torch.cdist(
        query.unsqueeze(dim=1), storage[entries]
    )
    target = target.square().squeeze(dim=1)

    # check
    assert torch.allclose(route.cpu(), indices)
    assert torch.allclose(
        distances.cpu(), target, atol=1e-3
    )

    #
    print('[PASS] test_routing_1()')


def main():
    test_routing_1()


if __name__ == '__main__':
    main()
