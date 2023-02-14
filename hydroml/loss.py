import torch
import torch.nn.functional as F

def MWSE(y_hat, y_true, weights, dim=([0], [1]), norm_weights=True):
    weights = weights.to(y_hat.device)
    square_err = (y_true - y_hat) ** 2
    weighted_se = torch.tensordot(weights, square_err, dims=dim) / len(weights)
    return torch.mean(weighted_se)


def DWSE(yhat, ytru, lmbda=8, loss_fun=F.mse_loss):
    loss = loss_fun(ytru, yhat)

    dx_tru = torch.diff(ytru, dim=-1)
    dx_hat = torch.diff(yhat, dim=-1)
    dx_loss = loss_fun(dx_tru, dx_hat)

    dy_tru = torch.diff(ytru, dim=-2)
    dy_hat = torch.diff(yhat, dim=-2)
    dy_loss = loss_fun(dy_tru, dy_hat)

    return loss + lmbda * (dx_loss + dy_loss)
