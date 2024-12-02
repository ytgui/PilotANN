import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path


def main():
    chunk_size = 64
    total_queries = 65536
    data_root = Path('.dataset/laion-1m')

    # load
    base = []
    for child in sorted(data_root.iterdir()):
        if child.suffix != '.npy':
            continue
        embedding = np.load(child.absolute())
        base.append(torch.FloatTensor(embedding))
    base = torch.cat(base, dim=0)

    # query
    query = torch.stack(
        random.choices(base, k=total_queries), dim=0
    )
    np.save('query.npy', arr=query)

    # target
    target = torch.zeros(
        [query.size(0), 100], dtype=torch.int32
    )
    for i in tqdm(range(0, query.size(0), chunk_size)):
        query_i = query[i:i + chunk_size]
        output = torch.topk(
            torch.cdist(query_i, base, p=2.0),
            k=100, dim=-1, largest=False, sorted=True
        )
        target[i:i + chunk_size] = output.indices
    np.save('groundtruth.npy', arr=target)

    #
    return


if __name__ == '__main__':
    main()
