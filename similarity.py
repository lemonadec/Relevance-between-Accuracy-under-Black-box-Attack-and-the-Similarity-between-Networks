# implements different similarity indexes
import torch
import numpy as np
from scipy.linalg import orth, norm


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


def LR(X, Y):
    """
    :param X: nxp1 tensor of activations of p1 neurons for n examples
    :param Y: nxp2 tensor of activations of p2 neurons for n examples
    :return: float, the linear regression index of X and Y

    When constructing an orthonormal basis for the range of A,
    we use orth() directly, which uses the SVD method
    Assume the features have been prepossessed to center the columns
    """
    X, Y = X.numpy(), Y.numpy()
    n = X.shape[0]
    QY = orth(Y.T)
    R2_LR = norm(QY.T @ X)**2 / norm(X)**2
    return R2_LR


def CCA(X, Y):
    """
    :param X: nxp1 tensor of activations of p1 neurons for n examples
    :param Y: nxp2 tensor of activations of p2 neurons for n examples
    :return: float, the CCA index of X and Y

    When constructing an orthonormal basis for the range of A,
    we use orth() directly, which uses the SVD method
    Assume the features have been prepossessed to center the columns
    """
    X, Y = X.numpy(), Y.numpy()
    n = X.shape[0]
    p1 = X.shape[1]
    QX, QY = orth(X.T), orth(Y.T)
    R2_CCA = norm(QY.T @ QX)**2 / p1
    return R2_CCA


def HSIC(X, Y):
    """
    :param X: nxp1 tensor of activations of p1 neurons for n examples
    :param Y: nxp2 tensor of activations of p2 neurons for n examples
    :return: float, the linear HSIC index of X and Y

    Assume the features have been prepossessed to center the columns
    """
    X, Y = X.numpy(), Y.numpy()
    n = X.shape[0]
    R_HSIC = norm(Y.T @ X) ** 2 / (n-1)**2
    return R_HSIC