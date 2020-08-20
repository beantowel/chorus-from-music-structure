import logging
import numpy as np
from copy import copy, deepcopy
from collections import defaultdict
from scipy.stats import mode
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt

from utility.common import *
from configs.modelConfigs import *
from configs.configs import DEBUG, logger


affinityPropagation = AffinityPropagation()


def modefilt(arr, kernel_size):
    dt = (kernel_size - 1) // 2
    newarr = np.zeros_like(arr, dtype=int)
    for i in range(len(arr)):
        window = arr[max(0, i - dt) : min(len(arr), i + dt + 1)]
        m, _ = mode(window)
        newarr[i] = m
    return newarr


def smoothCliques(cliques, size, kernel_size=SMOOTH_KERNEL_SIZE):
    # arr[<frames>]:<label>
    arr = np.zeros(size, dtype=int)
    for i, clique in enumerate(cliques):
        for idx in clique:
            arr[idx] = i
    while True:
        newarr = modefilt(arr, kernel_size)
        if (arr == newarr).all():
            break
        else:
            arr = newarr.copy()

    newCliques = cliquesFromArr(arr)
    return newCliques


def cliquesFromSSM(ssm_f, show=DEBUG):
    ssm = ssm_f[1] - np.max(ssm_f[1])
    labels = affinityPropagation.fit_predict(ssm)
    cliques = cliquesFromArr(labels)
    cliques = sorted(cliques, key=lambda c: c[0])
    if show:
        size = ssm.shape[0]
        mat = ssm
        printArray(ssm, "ssm")
        lssm = getLabeledSSM(cliques, size)
        _, axis = plt.subplots(1, 2)
        axis = axis.flatten()
        axis[0].imshow(mat)
        axis[1].imshow(lssm)
        plt.show()
    return cliques


def isAdjacent(cliqueA, cliqueB, dis=ADJACENT_DELTA_DISTANCE, dblock=0):
    _, x = filteredCliqueEnds(cliqueA)
    y, _ = filteredCliqueEnds(cliqueB)
    deltaX2y = np.array([np.min(np.abs(y - i)) for i in x])
    deltaY2x = np.array([np.min(np.abs(x - i)) for i in y])

    neighborX = np.sum(deltaX2y <= dis)
    neighborY = np.sum(deltaY2x <= dis)
    res = all(
        [
            neighborX > 0,
            neighborY > 0,
            len(y) - neighborY <= dblock,
            len(x) - neighborX <= dblock,
        ]
    )
    return res


def error(origCliques, mergedCliques, size, times, show=DEBUG):
    olssm = getLabeledSSM(origCliques, size)
    mlssm = getLabeledSSM(mergedCliques, size)
    olssm[olssm > 0] = 1
    mlssm[mlssm > 0] = 1
    # false negative + false positive
    fnerr = np.sum((mlssm == 0) & olssm) / (np.sum(olssm) + EPSILON)
    fperr = np.sum(mlssm & (olssm == 0)) / (np.sum(olssm == 0) + EPSILON)
    err = fnerr + max(0, fperr - FALSE_POSITIVE_ERROR)
    if show:
        logger.debug(
            f"errs={fnerr:.5f},{fperr:.5f} sum={err:.3f} len={len(mergedCliques)}"
        )
        x, xm = getLabeledSSM(origCliques, size), getLabeledSSM(mergedCliques, size)
        labels = [x[i, i] for i in range(size)]
        xm[xm > 0] = 10
        plt.imshow(x + xm)
        plt.plot(labels)
        plt.show()
    return err


def mergeAdjacentCliques(cliques, dis=ADJACENT_DELTA_DISTANCE, dblock=0):
    logger.debug(f"merge cliques, dis={dis} dblock={dblock}")
    size = len(cliques)
    adjMat = np.zeros([size] * 2, dtype=int)
    # calculate adjacency matrix
    for i in range(size):
        adjMat[i, i] = 1
        for j in range(i + 1, size):
            res = isAdjacent(cliques[i], cliques[j], dis=dis, dblock=dblock)
            adjMat[i, j] = res
            adjMat[j, i] = res
    # generate transitive closure
    for k in range(size):
        for i in range(size):
            for j in range(size):
                if adjMat[i, k] and adjMat[k, j]:
                    adjMat[i, j] = 1
    # merge cliques
    # key:smallest clique label in connected component
    # value:frame number list
    cliquesDic = defaultdict(list)
    for i in range(size):
        for j in range(size):
            if adjMat[i, j]:
                cliquesDic[j].extend(cliques[i])
                cliquesDic[j] = sorted(cliquesDic[j])
                break
    newCliques = list(cliquesDic.values())
    newCliques = sorted(newCliques, key=lambda c: c[0])
    return newCliques


def buildRecurrence(cliques, times):
    cliques = deepcopy(cliques)
    size = len(times) - 1
    mergedCliquesList = [
        smoothCliques(mergeAdjacentCliques(cliques, dis=dis, dblock=dblock), size)
        for dis in DELTA_DIS_RANGE
        for dblock in DELTA_BLOCK_RANGE
    ]
    # mclen = len(mergedCliquesList)
    # for i in range(1):
    #     mergedCliquesList.extend(
    #         [
    #             smoothCliques(mergeAdjacentCliques(cs), size)
    #             for cs in mergedCliquesList[-mclen:]
    #         ]
    #     )
    errors = [error(cliques, ncs, size, times) for ncs in mergedCliquesList]
    indices = np.argsort(errors)
    for i in indices:
        newCliques = mergedCliquesList[i]
        predicate = all([len(newCliques) >= MIN_STRUCTURE_COUNT,])
        if predicate:
            return newCliques
    logger.warn(f"seqrecur failed, cliqueLengths={[len(x) for x in mergedCliquesList]}")
    return mergedCliquesList[0]
