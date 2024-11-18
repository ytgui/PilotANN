import torch
import argparse
from pilot_ann import utils, proto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--top_k', type=int, default=10
    )
    parser.add_argument(
        '--n_queries', type=int, default=1024
    )
    parser.add_argument(
        '--dataset', type=str, default='deep-1m'
    )
    args = parser.parse_args()

    # init
    loader = utils.DataLoader(name=args.dataset)
    query, target = loader.load_query(
        n_queries=args.n_queries, k=args.top_k
    )
    storage = loader.load_storage()

    # index
    for index in [
        proto.IndexNSW(d_model=loader.d_model),
        proto.IndexOracle(
            d_model=loader.d_model, known_portion=1/2
        ),
        proto.IndexOracle(
            d_model=loader.d_model, known_portion=1/4
        ),
        proto.IndexOracle(
            d_model=loader.d_model, known_portion=1/8
        ),
        proto.IndexOracle(
            d_model=loader.d_model, known_portion=1/16
        ),
        proto.IndexOracle(
            d_model=loader.d_model, known_portion=1/24
        ),
        proto.IndexOracle(
            d_model=loader.d_model, known_portion=1/32
        )
    ]:
        # train
        print('=>', type(index).__name__,
              'd={}'.format(loader.d_model))
        index.train(storage)

        # search
        for ef_search in [16, 32, 64, 96, 128]:
            print('----------')
            print('ef_search:', ef_search)
            output = index.search(
                query, k=args.top_k, ef_search=ef_search
            )

            # stats
            for name in output:
                if name in ['topk', 'visited']:
                    continue
                print('{}@{}: {:.3f}'.format(
                    name, args.top_k, output[name]
                ))

            # score
            score = utils.recall(output['topk'], target=target)
            print('score@{}: {:.3f}'.format(args.top_k, score))


if __name__ == '__main__':
    main()
