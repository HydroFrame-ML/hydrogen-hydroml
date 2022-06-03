import torch

def MWSE(y_hat, y_true, weights, dim=([0], [1])):
    weights = weights.to(y_hat.device)
    square_err = (y_true - y_hat) ** 2
    weighted_se = torch.tensordot(weights, square_err, dims=dim) / len(weights)
    return torch.mean(weighted_se)
