import torch
import heapq
import random


def search_simple(indptr: list,
                  indices: list,
                  mapping: list,
                  storage: torch.FloatTensor,
                  query: torch.FloatTensor,
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
    while len(visited) < ef_search:
        u = random.choice(indices)
        if u in visited:
            continue
        visited.add(u)
        dist = torch.dist(
            query, storage[mapping[u]], p=2.0
        )
        heapq.heappush(openlist, [dist.item(), u])
        heapq.heappush(topk, [-dist.item(), u])

    # traverse
    while openlist:
        # shrink
        if len(openlist) > ef_search:
            openlist = [
                heapq.heappop(openlist)
                for _ in range(ef_search)
            ]

        # expand
        worst = -topk[0][0]
        dist, u = heapq.heappop(openlist)
        if len(topk) >= ef_search and dist >= worst:
            break

        # visit
        for v in indices[indptr[u]:indptr[u + 1]]:
            if v < 0:
                break
            if v in visited:
                continue
            visited.add(v)
            dist = torch.dist(
                query, storage[mapping[v]], p=2.0
            )
            heapq.heappush(openlist, [dist.item(), v])
            heapq.heappushpop(topk, [-dist.item(), v])

    # output
    while len(topk) > k:
        heapq.heappop(topk)
    topk = [node for _, node in topk]
    return topk
