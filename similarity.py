# implements different similarity indexes
import torch
import numpy as np
from scipy.linalg import orth, norm, svd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def CKA(x, y):
    """ calculating the CKA index of two feature
    Argument:
    x, y(torch.Tensor): n * d Tensor containing n feature vector of dimension d
    Return:
    CKA_result(float): CKA index between x and y."""
    n = x.shape[0]
    H = torch.eye(n, n) - 1 / n
    H = H.to(device)
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
    X, Y = X.cpu(), Y.cpu()
    X, Y = X.numpy(), Y.numpy()
    n = X.shape[0]
    QY = orth(Y)
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
    X, Y = X.cpu(), Y.cpu()
    X, Y = X.numpy(), Y.numpy()
    p1 = X.shape[1]
    QX, QY = orth(X), orth(Y)
    R2_CCA = norm(QY.T @ QX)**2 / p1
    return R2_CCA

def CCA_rou(X, Y):
    """
    :param X: nxp1 tensor of activations of p1 neurons for n examples
    :param Y: nxp2 tensor of activations of p2 neurons for n examples
    :return: float, the CCA index of X and Y
    
    similar to standard CCA, the only difference is the norm used in calculating R2_CCA
    """
    X, Y = X.numpy(), Y.numpy()
    n = X.shape[0]
    p1 = X.shape[1]
    QX, QY = orth(X), orth(Y)
    # nuclear norm
    _, cal_norm, _ = svd(QY.T @ QX)
    R2_CCA = cal_norm.sum() / p1
    return R2_CCA

def SVCCA(X, Y, epsilon = 0.98):
    """
    :param X: nxp1 tensor of activations of p1 neurons for n examples
    :param Y: nxp2 tensor of activations of p2 neurons for n examples
    :epsilon: float, threshold to truncate Tx and Ty
    :return: float, the linear regression index of X and Y
    
    here we use truncated Ux and Uy to represent UxTx and UyTy
    """
    X, Y = X.numpy(), Y.numpy()
    Ux, Sx, _ = svd(X)
    Sx = Sx.cumsum()/Sx.sum()
    for i,j in enumerate(Sx):
        if j>epsilon:
            tx=i+1
            break
    Ux = Ux[:,:tx]
    Uy, Sy, _ = svd(Y)
    Sy = Sy.cumsum()/Sy.sum()
    for i,j in enumerate(Sy):
        if j>epsilon:
            ty=i+1
            break
    Uy = Uy[:,:ty]
    cal_svcca = norm(Uy.T @ Ux)**2 / min(tx, ty)
    return cal_svcca

def SVCCA_rou(X, Y, epsilon = 0.98):
    """
    :param X: nxp1 tensor of activations of p1 neurons for n examples
    :param Y: nxp2 tensor of activations of p2 neurons for n examples
    :epsilon: float, threshold to truncate Tx and Ty
    :return: float, the linear regression index of X and Y
    
    similar to standard SVCCA, the only difference is the norm used in cal_svcca
    """
    X, Y = X.numpy(), Y.numpy()
    Ux, Sx, _ = svd(X)
    Sx = Sx.cumsum()/Sx.sum()
    for i,j in enumerate(Sx):
        if j>epsilon:
            tx=i+1
            break
    Ux = Ux[:,:tx]
    Uy, Sy, _ = svd(Y)
    Sy = Sy.cumsum()/Sy.sum()
    for i,j in enumerate(Sy):
        if j>epsilon:
            ty=i+1
            break
    Uy = Uy[:,:ty]
    _, cal_norm, _ = svd(Uy.T @ Ux)
    cal_svcca = cal_norm.sum() / min(tx, ty)
    return cal_svcca

def PWCCA(X, Y):
    X, Y = X.numpy(), Y.numpy()
    L_11 = X.T @ X
    L_12 = X.T @ Y
    L_22 = Y.T @ Y
    L_11_U, L_11_S, L_11_V = svd(L_11)
    L_22_U, L_22_S, L_22_V = svd(L_22)
    L_11_inv, L_22_inv = L_11_U@np.diag(L_11_S**(-1/2))@L_11_V, L_22_U@np.diag(L_22_S**(-1/2))@L_22_V
    L = L_11_inv @ L_12 @ L_22_inv
    U, R, V = svd(L)
    H = U.T @ L_11_inv @ X.T
    alpha_Q = abs(H @ X)
    alpha = np.array([alpha_Q[i,:].sum() for i in range(alpha_Q.shape[0])])
    cal_pwcca = 0
    c = min(X.shape[1], Y.shape[1])
    for i in range(c):
        cal_pwcca += alpha[i]*R[i]
    cal_pwcca /= alpha.sum()
    return cal_pwcca

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
    sample_ndim = 5000  # default
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
