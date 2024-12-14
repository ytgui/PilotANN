import argparse
from common import ANNProfiler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--top_k', type=int, default=10
    )
    parser.add_argument(
        '--method', type=str, default='faiss'
    )
    parser.add_argument(
        '--d_principle', type=int, default=96
    )
    parser.add_argument(
        '--sample_ratio', type=float, default=0.25
    )
    parser.add_argument(
        '--n_neighbors', type=int, default=32
    )
    parser.add_argument(
        '--n_queries', type=int, default=1024
    )
    parser.add_argument(
        '--router', type=str, default='router32x32'
    )
    parser.add_argument(
        '--algorithm', type=str, default='hnsw',
        choices=['hnsw', 'nsg']
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

    # train
    profiler.build(
        method=args.method, n_neighbors=args.n_neighbors
    )
    if args.method == 'faiss':
        checkpoint = '{}.faiss'.format(args.dataset)
    elif args.method == 'search-pilot':
        checkpoint = '{}.ckpt'.format(args.dataset)
    else:
        raise NotImplementedError
    profiler.dump(checkpoint=checkpoint)
    print('done')


if __name__ == '__main__':
    main()
