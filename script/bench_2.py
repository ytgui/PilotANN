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
    parser.add_argument(
        '--dataset', type=str, default='deep-1m'
    )
    args = parser.parse_args()

    # init
    profiler = ANNProfiler(
        dataset=args.dataset,
        algorithm=args.algorithm,
        n_queries=args.n_queries,
        entry_method=None,
        sample_ratio=None,
        d_principle=None
    )

    # evaluate
    records = {
        'search-pilot': [], 'faiss': []
    }
    for method in records.keys():
        profiler.load(
            checkpoint=args.checkpoint
        )
        for ef_search in args.ef_search:
            time.sleep(2.0)
            recall, timing = profiler.run(
                k=args.top_k, ef_search=ef_search
            )
            print('{}: recall={:.3f}, duration={:.3f}ms'
                  .format(method, recall, timing))
            records[method].append({
                'recall': recall,
                'timing': timing,
                'ef_search': ef_search
            })

    # print
    for method in records.keys():
        for record in records[method]:
            print(method, record)


if __name__ == '__main__':
    main()
