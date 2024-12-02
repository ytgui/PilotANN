import math
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_fuzz(name: str):
    n_queries = 4096

    #
    if name.startswith('fuzz'):
        d_model = random.choice(
            [64, 96, 128, 200, 512, 768]
        )
        if name.endswith('16k'):
            n_storage = random.randint(
                16384 // 2, 2 * 16384
            )
        elif name.endswith('64k'):
            n_storage = random.randint(
                65536 // 2, 2 * 65536
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    #
    query = 2.0 * torch.sub(
        torch.rand(size=[n_queries, d_model]), 0.5
    )
    storage = 2.0 * torch.sub(
        torch.rand(size=[n_storage, d_model]), 0.5
    )
    target = torch.topk(
        torch.cdist(query, storage, p=2.0),
        k=100, dim=-1, largest=False
    )
    return query, storage, target.indices


def read_bins(filename: str, dtype: np.dtype):
    vec = np.fromfile(
        filename, dtype=dtype
    )
    num = vec[0].view(np.int32)
    dim = vec[1].view(np.int32)
    vec = vec[2:].reshape(num, dim)
    return vec


def search_target(query: torch.Tensor,
                  storage: torch.Tensor,
                  chunk_size: int = 64,
                  top_k: int = 100):
    if len(storage) <= 64 * 1024:
        topk = torch.topk(
            torch.cdist(query, storage, p=2.0),
            k=top_k, dim=-1, largest=False, sorted=True
        )
        target = topk.indices
    else:
        target = torch.zeros(
            [query.size(0), top_k], dtype=torch.long
        )
        n_chunks = math.ceil(len(query) / chunk_size)
        for i in tqdm(range(n_chunks), desc='init dataset'):
            left = i * chunk_size
            right = left + chunk_size
            output = torch.topk(
                torch.cdist(
                    query[left:right], storage, p=2.0
                ),
                k=top_k, dim=-1, largest=False, sorted=True
            )
            target[left:right] = output.indices
    return target


def shrink_dataset(query: torch.Tensor,
                   storage: torch.Tensor,
                   n_targets: int):
    indices = torch.randint(
        high=storage.size(0), size=[n_targets]
    )
    storage = torch.index_select(
        storage, dim=0, index=indices
    )
    target = search_target(
        query=query, storage=storage
    )
    return storage, target


def load_deep(name: str):
    if name == 'deep-64k':
        n_storage = 65536
        data_root = '.dataset/deep-1m'
    elif name == 'deep-1m':
        n_storage = 1_000_000
        data_root = '.dataset/deep-1m'
    elif name == 'deep-10m':
        n_storage = 10_000_000
        data_root = '.dataset/deep-10m'
    elif name == 'deep-50m':
        n_storage = 50_000_000
        data_root = '.dataset/deep-100m'
    elif name == 'deep-100m':
        n_storage = 100_000_000
        data_root = '.dataset/deep-100m'
    elif name == 'text2img-64k':
        n_storage = 65536
        data_root = '.dataset/text2img-1m'
    elif name == 'text2img-1m':
        n_storage = 1_000_000
        data_root = '.dataset/text2img-1m'
    elif name == 'text2img-10m':
        n_storage = 10_000_000
        data_root = '.dataset/text2img-50m'
    elif name == 'text2img-50m':
        n_storage = 50_000_000
        data_root = '.dataset/text2img-50m'
    else:
        raise NotImplementedError

    # query
    query = read_bins(
        data_root + '/query.fbin', dtype=np.float32
    )
    storage = read_bins(
        data_root + '/base.fbin', dtype=np.float32
    )
    query = torch.from_numpy(query).contiguous()
    storage = torch.from_numpy(storage).contiguous()

    # target
    target = None
    try:
        target = read_bins(
            data_root + '/groundtruth.ibin', dtype=np.int32
        )
        target = torch.from_numpy(target).contiguous()
    except FileNotFoundError:
        pass

    # shrink
    if storage.size(0) > n_storage:
        storage, target = shrink_dataset(
            query, storage=storage, n_targets=n_storage
        )
    if target is None:
        target = search_target(
            query=query, storage=storage
        )
    return query, storage, target


def load_numpy(name: str):
    if name == 'laion-64k':
        n_storage = 65536
        data_root = '.dataset/laion-1m'
    elif name == 'laion-1m':
        n_storage = 1_000_000
        data_root = '.dataset/laion-1m'
    else:
        raise NotImplementedError

    # query
    query = torch.FloatTensor(
        np.load(data_root + '/query.npy')
    )
    target = torch.LongTensor(
        np.load(data_root + '/groundtruth.npy')
    )

    # load
    storage = []
    for child in sorted(Path(data_root).iterdir()):
        if child.suffix != '.npy':
            continue
        if child.name in ['query.npy',
                          'groundtruth.npy']:
            continue
        embedding = torch.FloatTensor(
            np.load(child.absolute())
        )
        storage.append(embedding)
    storage = torch.cat(storage, dim=0)

    # shrink
    if storage.size(0) > n_storage:
        storage, target = shrink_dataset(
            query, storage=storage, n_targets=n_storage
        )
    return query, storage, target


class DataLoader:
    def __init__(self, name: str):
        if name.startswith('fuzz'):
            output = load_fuzz(name)
        elif name.startswith('deep'):
            output = load_deep(name)
        elif name.startswith('text2img'):
            output = load_deep(name)
        elif name.startswith('wiki'):
            output = load_numpy(name)
        elif name.startswith('laion'):
            output = load_numpy(name)
        else:
            raise NotImplementedError
        #
        self.query: torch.Tensor = output[0]
        self.storage: torch.Tensor = output[1]
        self.target: torch.Tensor = output[2]

    @property
    def d_model(self):
        return self.query.size(-1)

    @property
    def n_storage(self):
        return self.storage.size(0)

    def load_storage(self):
        return self.storage

    def load_query(self, n_queries: int, k: int):
        if k > self.target.size(-1):
            raise RuntimeError
        indices = torch.randint(
            high=len(self.query), size=[n_queries]
        )
        query = torch.index_select(
            self.query, dim=0, index=indices
        )
        target = torch.index_select(
            self.target, dim=0, index=indices
        )
        return query, target[:, :k]
