import time
import argparse
from common import ANNProfiler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--top_k', type=int, default=10
    )
    parser.add_argument(
        '--d_principle', type=int, default=64
    )
    parser.add_argument(
        '--sample_ratio', type=float, default=0.5
    )
    parser.add_argument(
        '--ef_search', type=list,
        default=[16, 32, 48, 64, 96, 128, 192, 256]
    )
    parser.add_argument(
        '--n_neighbors', type=int, default=32
    )
    parser.add_argument(
        '--n_queries', type=int, default=1024
    )
    parser.add_argument(
        '--algorithm', type=str, default='hnsw',
        choices=['hnsw', 'nsg']
    )
    parser.add_argument(
        '--router', type=str, default='router32x32'
    )
    parser.add_argument(
        '--dataset', type=str, default='deep-1m'
    )
    args = parser.parse_args()

    # init
    profiler = ANNProfiler(
        dataset=args.dataset,
        algorithm=args.algorithm,
        entry_method=args.router,
        sample_ratio=args.sample_ratio,
        d_principle=args.d_principle,
        n_queries=args.n_queries
    )

    # evaluate
    records = {
        'faiss': [], 'search-pilot': []
    }
    for method in records.keys():
        profiler.build(
            method=method, n_neighbors=args.n_neighbors
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
                'n_neighbors': args.n_neighbors,
                'ef_search': ef_search
            })

    # print
    for method in records.keys():
        for record in records[method]:
            print(method, record)


if __name__ == '__main__':
    main()
