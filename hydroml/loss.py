import torch

def MWSE(y_hat, y_true, weights, dim=([0], [1]), norm_weights=True):
    weights = weights.to(y_hat.device)
    square_err = (y_true - y_hat) ** 2
    weighted_se = torch.tensordot(weights, square_err, dims=dim) / len(weights)
    return torch.mean(weighted_se)


def DWSE(y_hat, y_true, lmda, loss_func):
    std_loss = loss_func(y_hat, y_true)
    
    dx_true = y_true[...,0:-1]-y_true[...,1:]
    dy_true = y_true[...,0:-1,:] - y_true[...,1:,:]
        
    dx_pred = y_hat[...,0:-1]-y_hat[...,1:]
    dy_pred = y_hat[...,0:-1,:] - y_hat[...,1:,:]

    dx_loss = torch.mean(lmda*(abs(dx_pred-dx_true))) #This result is a grid of dx diff
    dy_loss = torch.mean(lmda*(abs(dy_pred-dy_true))) #this result is a grid of dy diff

    deriv_loss = std_loss + dx_loss + dy_loss

    return deriv_loss
