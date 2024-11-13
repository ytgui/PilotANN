import torch
from pilot_ann import utils, kernels


def evaluate_cuda(graph_type: str, sampling: bool):
    k = 10
    ef_search = 64
    beam_width = 4
    n_neighbors = 32
    batch_size = 1024
    cuda_device = 'cuda'

    # init
    loader = utils.DataLoader('fuzz-64k')
    storage = loader.load_storage()
    query, target = loader.load_query(
        n_queries=batch_size, k=k
    )
    distances = torch.cdist(query, storage, p=2.0)

    # graph
    indptr, indices = utils.graph_init(
        storage, graph_type, n_neighbors=n_neighbors
    )
    mapping = list(range(loader.n_storage))

    # sampling
    if sampling:
        subgraph = utils.subgraph_init(
            indptr=indptr, indices=indices,
            storage=storage, graph_type=graph_type,
            n_samples=loader.n_storage // 4,
            n_neighbors=n_neighbors
        )
        indptr, indices, nodelist, mapping = subgraph
        assert max(mapping) + 1 == len(nodelist)
        assert len(set(mapping)) == len(nodelist) + 1
        assert all(
            mapping[x] != -1 for x in nodelist
        )
        storage = storage[nodelist]

    # traverse
    query = query.to(device=cuda_device)
    storage = storage.to(device=cuda_device)
    distances = distances.to(device=cuda_device)
    indptr = torch.LongTensor(indptr).to(cuda_device)
    indices = torch.LongTensor(indices).to(cuda_device)
    mapping = torch.LongTensor(mapping).to(cuda_device)
    output_I = torch.empty(
        size=[batch_size, beam_width, ef_search],
        dtype=torch.long, device=cuda_device
    )
    output_D = torch.empty(
        size=[batch_size, beam_width, ef_search],
        dtype=torch.float, device=cuda_device
    )
    initial_I = torch.randint(
        high=loader.n_storage, size=[batch_size, ef_search],
        device=cuda_device
    )
    initial_D = torch.gather(
        distances.square(), dim=-1, index=initial_I
    )
    kernels.traverse_cuda(
        output_I=output_I, output_D=output_D,
        indptr=indptr, indices=indices, mapping=mapping,
        storage=storage, query=query, initial_I=initial_I,
        initial_D=initial_D, n_neighbors=n_neighbors
    )
    output_I = output_I.flatten(start_dim=1).cpu()
    output_D = output_D.flatten(start_dim=1).cpu()

    # top-k
    orders = torch.topk(
        output_D, k=k, dim=-1, largest=False
    )
    topk_I = torch.gather(
        output_I, dim=-1, index=orders.indices
    )
    score = utils.recall(topk_I, target=target)
    print('{}: d_model={}, recall@{}: {:.3f}'.format(
        graph_type, loader.d_model, k, score
    ))

    # check recall
    thresholds = {
        128: 0.60, 512: 0.35, 1024: 0.20
    }
    if sampling:
        thresholds = {
            k: v / 4.0 for k, v in thresholds.items()
        }
    assert score >= next(
        v for k, v in thresholds.items() if loader.d_model <= k
    )

    # check distance
    target = torch.gather(
        distances.cpu(), dim=-1, index=output_I
    )
    assert torch.allclose(
        output_D, target.square(), atol=1e-3
    )


def test_traverse_cuda_1():
    for graph_type in ['nsg', 'nsw']:
        evaluate_cuda(
            graph_type=graph_type, sampling=False
        )

    #
    print('[PASS] test_traverse_cuda_1()')


def test_traverse_cuda_2():
    for graph_type in ['nsg', 'nsw']:
        evaluate_cuda(
            graph_type=graph_type, sampling=True
        )

    #
    print('[PASS] test_traverse_cuda_2()')


def main():
    test_traverse_cuda_1()
    test_traverse_cuda_2()


if __name__ == '__main__':
    main()
