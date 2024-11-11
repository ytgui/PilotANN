import torch


def recall(predict: torch.Tensor,
           target: torch.Tensor):
    if not isinstance(predict, torch.Tensor):
        raise RuntimeError
    if not isinstance(target, torch.Tensor):
        raise RuntimeError
    if predict.size() != target.size():
        raise RuntimeError
    assert target.dim() == 2

    #
    recalls = []
    for y_hat, y in zip(predict, target):
        y = set(y.tolist())
        y_hat = set(y_hat.tolist())
        inter = set(y).intersection(y_hat)
        recalls.append(len(inter) / len(y))

    #
    return sum(recalls) / len(recalls)
