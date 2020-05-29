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


# dists2plane, removezeros, find_max_match, maxmatch 全都是wlw的index的函数，要求similarity，只需要调用maxmatch函数就可以了

_ZERO = 1e-16


def dists2plane(X, Y):
    """
    Calucating the distances of a group of vectors to a plane
    Assume norm is large enough
    :param X: [N1, D]
    :param Y: [N2, D]
    :return: [N1,]
    """
    Y_t = np.transpose(Y)
    X_t = np.transpose(X)
    solution = np.linalg.lstsq(Y_t, X_t, rcond=None)
    dist = np.linalg.norm(np.dot(Y_t, solution[0]) - X_t, axis=0)
    norm = np.linalg.norm(X_t, axis=0)
    return dist / norm


def remove_zeros(X):
    """
    Remove zero-norm vectors
    Args:
        X: [N, D]
    Returns:
        non-zero vectors: [N',]
        non-zero indices: [N',]
    """
    assert X.ndim == 2, "Only support 2-D X"
    norm_X = np.linalg.norm(X, axis=1)
    non_zero = np.where(norm_X > _ZERO)[0]
    return X[non_zero], non_zero


def find_maximal_match(X, Y, eps, has_purge=False):
    """
    Find maximal match set between X and Y
    Args:
        X: [N1, D]
        Y: [N2, D]
        eps: scalar
        has_purge: whether X and Y have removed zero vectors
    Returns:
        idx_X: X's match set indices
        idx_Y: Y's match set indices
    """
    assert X.ndim == 2 and Y.ndim == 2, 'Check dimensions of X and Y'
    # if _DEBUG: print('eps={:.4f}'.format(eps))

    if not has_purge:
        X, non_zero_X = remove_zeros(X)
        Y, non_zero_Y = remove_zeros(Y)

    idx_X = np.arange(X.shape[0])
    idx_Y = np.arange(Y.shape[0])

    if len(idx_X) == 0 or len(idx_Y) == 0:
        return idx_X[[]], idx_Y[[]]

    flag = True
    while flag:
        flag = False

        # tic = time.time()
        dist_X = dists2plane(X[idx_X], Y[idx_Y])
        # toc = time.time()
        # print(toc-tic)
        remain_idx_X = idx_X[dist_X <= eps]

        if len(remain_idx_X) < len(idx_X):
            flag = True

        idx_X = remain_idx_X
        if len(idx_X) == 0:
            idx_Y = idx_Y[[]]
            break

        # tic = time.time()
        dist_Y = dists2plane(Y[idx_Y], X[idx_X])
        # toc = time.time()
        # print(toc-tic)
        remain_idx_Y = idx_Y[dist_Y <= eps]

        if len(remain_idx_Y) < len(idx_Y):
            flag = True

        idx_Y = remain_idx_Y
        if len(idx_Y) == 0:
            idx_X = idx_X[[]]
            break

        # if _DEBUG: print('|X|={:d}, |Y|={:d}'.format(len(idx_X), len(idx_Y)))

    if not has_purge:
        idx_X = non_zero_X[idx_X]
        idx_Y = non_zero_Y[idx_Y]

    return idx_X, idx_Y


def maxmatch(mat0, mat1, epsilon=0.5):
    # fix random seed
    np.random.seed(0)
    # aliases
    nb_samples = 128  # batch size=128
    sample_ndim = 10000  # default
    sample_iter = 16  # default

    # reshape
    mat0 = mat0.numpy()  # 原来是tensor
    mat1 = mat1.numpy()

    mat0 = mat0[:nb_samples, ...]
    mat1 = mat1[:nb_samples, ...]

    # [N, C, H, W] -> [C, N, H, W]
    mat0 = mat0.transpose([1, 0, 2, 3])
    mat1 = mat1.transpose([1, 0, 2, 3])
    mat0 = mat0.reshape([mat0.shape[0], -1])
    mat1 = mat1.reshape([mat1.shape[0], -1])
    assert mat0.shape[1] == mat1.shape[1], 'Check the sizes of two sets'

    # tic = time.time()
    ms = 0
    for iter in range(sample_iter):
        sample_idx = np.random.choice(mat0.shape[1], sample_ndim, replace=False)
        X = mat0[:, sample_idx]
        Y = mat1[:, sample_idx]

        idx_X, idx_Y = find_maximal_match(X, Y, epsilon)
        mms = float(len(idx_X) + len(idx_Y)) / (len(X) + len(Y))
        ms = ms + mms

    ms = ms / sample_iter  # 这里稍微改了一点论文里的东西，相当于把激活向量映射到一个10000维的子空间上，再去看他们是否有match（不然太大了）

    # toc = time.time()

    return ms
