# implements different similarity indexes
import torch


def CKA(x, y):
    """ calculating the CKA index of two feature
    Argument:
    x, y(torch.Tensor): n * d Tensor containing n feature vector of dimension d
    Return:
    CKA_result(float): CKA index between x and y."""
    H = torch.eye(x.shape(0), x.shape(0))
    