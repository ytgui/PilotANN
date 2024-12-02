from pilot_ann import utils


def test_loader():
    k = 10
    batch_size = 1024

    #
    for name in [
        'fuzz-16k', 'fuzz-64k',
        'deep-64k', 'text2img-64k', 'laion-64k',
        'deep-1m', 'text2img-1m', 'laion-1m'
    ]:
        # init
        loader = utils.DataLoader(name=name)

        # load
        storage = loader.load_storage()
        query, target = loader.load_query(batch_size, k=k)
        print('[{}] storage={}, query={}, target={}'.format(
            name, storage.size(), query.size(), target.size()
        ))

        # check
        assert query.size(0) == batch_size
        assert target.size(-1) == k

    #
    print('[PASS] test_loader()')


def main():
    test_loader()


if __name__ == "__main__":
    main()
