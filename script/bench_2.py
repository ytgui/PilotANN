import time
import argparse
from common import ANNProfiler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--top_k', type=int, default=10
    )
    parser.add_argument(
        '--ef_search', type=list,
        default=[16, 32, 48, 64, 96, 128, 192, 256]
    )
    parser.add_argument(
        '--algorithm', type=str, default='hnsw',
        choices=['hnsw', 'nsg']
    )
    parser.add_argument(
        '--n_queries', type=int, default=1024
    )
    parser.add_argument(
        '--checkpoint', type=str, default='deep-1m.faiss'
    )
    args = parser.parse_args()

    # init
    dataset, method = str.split(
        args.checkpoint, sep='.', maxsplit=-1
    )
    profiler = ANNProfiler(
        dataset=dataset,
        algorithm=args.algorithm,
        n_queries=args.n_queries,
        entry_method=None,
        sample_ratio=None,
        d_principle=None
    )

    # evaluate
    records = []
    profiler.load(
        '.index/{}'.format(args.checkpoint)
    )
    for ef_search in args.ef_search:
        time.sleep(2.0)
        recall, timing = profiler.run(
            k=args.top_k, ef_search=ef_search
        )
        print('{}: recall={:.3f}, duration={:.3f}ms'
              .format(method, recall, timing))
        records.append({
            'recall': recall,
            'timing': timing,
            'ef_search': ef_search
        })
    print(method, records)


if __name__ == '__main__':
    main()
