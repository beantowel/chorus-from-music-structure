import logging
import numpy as np
from copy import copy, deepcopy
from collections import defaultdict
from scipy.stats import mode
from scipy.sparse.csgraph import floyd_warshall
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product

from utility.common import cliquesFromArr, filteredCliqueEnds, getLabeledSSM, printArray
from configs.modelConfigs import (
    ADJACENT_DELTA_DISTANCE,
    DELTA_DIS_RANGE,
    EPSILON,
    FALSE_POSITIVE_ERROR,
    MIN_STRUCTURE_COUNT,
    SMOOTH_KERNEL_SIZE,
    SMOOTH_KERNEL_SIZE_RANGE,
)
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


# def cliquesFromSSM(ssm_f, show=False):
#     def selectClique(cliques):
#         lens = [len(c) for c in cliques]
#         groupCounts = [len(filteredCliqueEnds(c, gap=10)[0]) for c in cliques]
#         ind = np.lexsort((lens, groupCounts))
#         return cliques[ind[-1]]

#     CLIQUE_CANDIDATES_COUNT = 7000
#     ssm = ssm_f[1]
#     size = ssm.shape[-1]

#     threshSSM = np.zeros_like(ssm, dtype=np.uint8)
#     for i, j in product(range(size), range(size)):
#         threshSSM[i, j] = 1 if ssm[i, j] >= SSM_LOG_THRESH else 0

#     g = nx.from_numpy_matrix(threshSSM)
#     gShow = deepcopy(g)
#     cliques = []
#     while g:
#         candidates = list(
#             [c for i, c in zip(range(CLIQUE_CANDIDATES_COUNT), nx.find_cliques(g))]
#         )
#         c = selectClique(candidates)

#         g.remove_nodes_from(c)
#         cliques.insert(0, sorted(list(c)))
#         if show:
#             if len(cliques) % 10 == 0:
#                 mat = nx.to_numpy_matrix(gShow) * 5
#                 mat += getLabeledSSM(cliques, size)
#                 plt.imshow(mat)
#                 plt.show()
#     cliques = sorted(cliques, key=lambda c: c[0])
#     return cliques


def cliquesFromSSM(ssm_f, show=False):
    # affinity propagation
    ssm = ssm_f[1] - np.max(ssm_f[1])
    median = np.median(ssm)
    for i in range(ssm.shape[0]):
        ssm[i, i] = median
    labels = affinityPropagation.fit_predict(ssm)
    # convert to cliques
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


def error(origCliques, mergedCliques, size, times, show=False):
    olssm = getLabeledSSM(origCliques, size)
    mlssm = getLabeledSSM(mergedCliques, size)
    olssm[olssm > 0] = 1
    mlssm[mlssm > 0] = 1
    # false negative + false positive
    fnerr = np.sum((mlssm == 0) & olssm) / (np.sum(olssm) + EPSILON)
    fperr = np.sum(mlssm & (olssm == 0)) / (np.sum(olssm == 0) + EPSILON)
    err = fnerr + max(0, fperr - FALSE_POSITIVE_ERROR)
    logger.debug(f"errs={fnerr:.5f},{fperr:.5f} sum={err:.3f} len={len(mergedCliques)}")
    if show:
        x, xm = getLabeledSSM(origCliques, size), getLabeledSSM(mergedCliques, size)
        labels = [x[i, i] for i in range(size)]
        xm[xm > 0] = 10
        plt.imshow(x + xm)
        plt.plot(labels)
        plt.show()
    return err


def mergeFind(adjLists, size):
    def getLabel(j):
        if labels[j] == j:
            return j
        else:
            labels[j] = getLabel(labels[j])
            return labels[j]

    labels = np.arange(size)
    for i in range(size):
        for j in adjLists[i]:
            labels[i] = getLabel(j)
    for i in range(size):
        labels[i] = getLabel(i)
    return labels


def mergeAdjacentCliques(cliques, dis=ADJACENT_DELTA_DISTANCE, dblock=0):
    logger.debug(f"merge cliques, dis={dis} dblock={dblock}")
    size = len(cliques)
    adjLists = [[] for i in range(size)]  # i < j: adjLists[j] = [..., i, ...]
    # calculate adjacency matrix
    for i in range(size):
        for j in range(i + 1, size):
            if isAdjacent(cliques[i], cliques[j], dis=dis, dblock=dblock):
                adjLists[j].append(i)
    # merge cliques in transitive closure
    # key:smallest clique label in connected component
    # value:frame number list
    cliquesDic = defaultdict(list)
    labels = mergeFind(adjLists, size)
    for i in range(size):
        cliquesDic[labels[i]].extend(cliques[i])

    newCliques = list(cliquesDic.values())
    newCliques = sorted(newCliques, key=lambda c: c[0])
    return newCliques


def buildRecurrence(cliques, times):
    logger.debug(f"build recurrence")
    cliques = deepcopy(cliques)
    size = len(times) - 1
    mergedCliquesList = [
        smoothCliques(
            mergeAdjacentCliques(cliques, dis=dis, dblock=dblock),
            size,
            kernel_size=kernelSize,
        )
        for dis in DELTA_DIS_RANGE
        for kernelSize in SMOOTH_KERNEL_SIZE_RANGE
        for dblock in [0, 1, 2]
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
        predicate = all(
            [
                len(newCliques) >= MIN_STRUCTURE_COUNT,
            ]
        )
        if predicate:
            return newCliques
    logger.warn(f"seqrecur failed, cliqueLengths={[len(x) for x in mergedCliquesList]}")
    return mergedCliquesList[0]
