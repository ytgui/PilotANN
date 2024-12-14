import torch
from pilot_ann import utils


def test_svd_1():
    d_model = 64
    n_embeddings = 1_000_000

    # init
    x = torch.randn(
        [n_embeddings, d_model]
    )
    y_1, V_1 = utils.svd(
        x, max_size=n_embeddings
    )
    y_2, V_2 = utils.svd(
        x, max_size=n_embeddings // 10
    )

    # check
    x_1 = torch.matmul(y_1, V_1.T)
    x_2 = torch.matmul(y_2, V_2.T)
    assert torch.allclose(x, x_1, atol=1e-3)
    assert torch.allclose(x, x_2, atol=1e-3)

    #
    print('[PASS] test_svd_1()')


def main():
    test_svd_1()


if __name__ == '__main__':
    main()
