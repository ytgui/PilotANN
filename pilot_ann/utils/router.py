import torch
import random


def router_init(storage: torch.Tensor,
                init_nodes: list,
                n_routers: int,
                n_leaves: int):
    assert storage.dim() == 2
    n_storage = storage.size(0)

    # layer-0
    router = set(
        [random.randrange(n_storage)]
    )
    while len(router) < n_routers:
        nodes = [
            random.randrange(n_storage)
            for _ in range(n_routers)
        ]
        dists = torch.cdist(
            storage[nodes], storage[list(router)], p=2.0
        )
        indice = torch.argmax(torch.sum(dists, dim=-1))
        router.add(nodes[indice])
    route_vectors = storage[list(router)]

    # layer-1
    visited = set()
    entry_nodes = [
        [] for _ in range(n_routers)
    ]
    for _ in range(n_leaves):
        # nodes
        nodes = set()
        while len(nodes) < min(1024, n_routers ** 2):
            if init_nodes:
                new_node = init_nodes.pop(-1)
            else:
                new_node = random.randrange(n_storage)
            if new_node in visited:
                continue
            visited.add(new_node)
            nodes.add(new_node)
        nodes = list(nodes)
        # distances
        dists = torch.cdist(
            route_vectors, storage[nodes], p=2.0
        )
        indices = torch.argmin(dists, dim=-1)
        for i in range(n_routers):
            entry_nodes[i].append(
                nodes[indices[i].item()]
            )
    entry_nodes = torch.LongTensor(entry_nodes)

    #
    return route_vectors, entry_nodes
