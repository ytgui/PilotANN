import torch
import heapq
import random


def search_simple(indptr: list,
                  indices: list,
                  mapping: list,
                  storage: torch.FloatTensor,
                  query: torch.FloatTensor,
                  initial: list,
                  k: int, ef_search: int):
    assert query.dim() == 1
    assert storage.dim() == 2
    assert query.size(-1) == storage.size(-1)
    assert isinstance(indptr, (list, tuple))
    assert isinstance(indices, (list, tuple))
    assert isinstance(mapping, (list, tuple))
    assert len(mapping) == len(indptr) - 1

    # init
    visited = set()
    openlist, topk = [], []

    # entry
    while True:
        if initial:
            u = initial.pop(-1)
        elif len(openlist) >= ef_search:
            break
        else:
            u = random.choice(indices)
        #
        if u in visited:
            continue
        visited.add(u)
        dist = torch.dist(
            query, storage[mapping[u]], p=2.0
        )
        heapq.heappush(openlist, [dist.item(), u])
        heapq.heappush(topk, [-dist.item(), u])

    # traverse
    for step in range(2 * ef_search):
        neighbors = []

        # expand
        if not openlist:
            break
        dist, u = heapq.heappop(openlist)
        if dist >= -topk[0][0]:
            break
        for v in indices[
            indptr[u]:indptr[u + 1]
        ]:
            if v < 0 or v in visited:
                continue
            neighbors.append(v)
            visited.add(v)

        # heapify
        for v in neighbors:
            dist = torch.dist(
                query, storage[mapping[v]], p=2.0
            )
            heapq.heappush(openlist, [dist.item(), v])
            heapq.heappushpop(topk, [-dist.item(), v])

        # shrink
        if len(openlist) > ef_search:
            openlist = [
                heapq.heappop(openlist)
                for _ in range(ef_search)
            ]

    # topk
    while len(topk) > k:
        heapq.heappop(topk)
    topk = [node for _, node in topk]

    # output
    output = {
        'topk': topk,
        'n_steps': step,
        'n_visited': len(visited)
    }
    return output
