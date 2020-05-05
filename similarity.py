# implements different similarity indexes
import torch


def CKA(x, y):
    """ calculating the CKA index of two feature
    Argument:
    x, y(torch.Tensor): n * d Tensor containing n feature vector of dimension d
    Return:
    CKA_result(float): CKA index between x and y."""
    n = x.shape(0)
    H = torch.eye(n, n) - 1 / n
    X, Y = x @ x.T, y @ y.T
    CKA_result = torch.trace(X @ H @ Y @ H) / \
        torch.sqrt(torch.trace(X @ H @ X @ H) * torch.trace(Y @ H @ Y @ H))
    return CKA_result
