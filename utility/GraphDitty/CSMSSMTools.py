"""
Programmer: Chris Tralie, 12/2016 (ctralie@alumni.princeton.edu)
Purpose: To provide tools for quickly computing all pairs self-similarity
and cross-similarity matrices
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from copy import copy
from scipy import sparse


def getCSM(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    :param X: An Mxd matrix holding the coordinates of M points
    :param Y: An Nxd matrix holding the coordinates of N points
    :return D: An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X ** 2, 1)[:, None] + np.sum(Y ** 2, 1)[None, :] - 2 * X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)


def getCSMEMD1D(X, Y):
    """
    An approximation of all pairs Earth Mover's 1D Distance
    """
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    XC = np.cumsum(X, 1)
    YC = np.cumsum(Y, 1)
    D = np.zeros((M, N))
    for k in range(K):
        xc = XC[:, k]
        yc = YC[:, k]
        D += np.abs(xc[:, None] - yc[None, :])
    return D


def getCSMCosine(X, Y):
    XNorm = np.sqrt(np.sum(X ** 2, 1))
    XNorm[XNorm == 0] = 1
    YNorm = np.sqrt(np.sum(Y ** 2, 1))
    YNorm[YNorm == 0] = 1
    D = (X / XNorm[:, None]).dot((Y / YNorm[:, None]).T)
    D = 1 - D  # Make sure distance 0 is the same and distance 2 is the most different
    return D


def getShiftInvariantCSM(metricFunc, wins_per_block=20, n_seq=1):
    def fun(X, Y):
        # X[m, d_feature * wins]
        m, d = X.shape
        d_feature = d // wins_per_block // n_seq
        # Xr[m, wins, d_feature, n_seq]
        Xr = copy(X.reshape(m, wins_per_block, d_feature, n_seq))
        DList = []
        for shift in range(d_feature):
            Xn = np.roll(Xr, shift, axis=2).reshape(m, d)
            D = metricFunc(Xn, Y)
            DList.append(D)
        DList = np.array(DList)
        res = np.min(DList, axis=0)
        return res

    return fun


def CSMToBinary(D, Kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix
    If Kappa = 0, take all neighbors
    If Kappa < 1 it is the fraction of mutual neighbors to consider
    Otherwise Kappa is the number of mutual neighbors to consider
    """
    N = D.shape[0]
    M = D.shape[1]
    if Kappa == 0:
        return np.ones((N, M))
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa * M))
    else:
        NNeighbs = Kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = sparse.coo_matrix((V, (I, J)), shape=(N, M))
    return ret.toarray()


def CSMToBinaryMutual(D, Kappa):
    """
    Take the binary AND between the nearest neighbors in one direction
    and the other
    """
    B1 = CSMToBinary(D, Kappa)
    B2 = CSMToBinary(D.T, Kappa).T
    return B1 * B2


def CSM2CRPEps(CSM, eps):
    """
    Convert a CSM to a cross-recurrence plot with an epsilon threshold
    :param CSM: MxN cross-similarity matrix
    :param eps: Cutoff epsilon
    :returns CRP: MxN cross-recurrence plot
    """
    CRP = np.zeros(CSM.shape)
    CRP[CSM <= eps] = 1
    return CRP
