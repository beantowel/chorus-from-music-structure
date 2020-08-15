import numpy as np
import networkx as nx
from copy import copy, deepcopy
from itertools import product
from collections import defaultdict
from queue import Queue
from scipy.stats import mode
from scipy.signal import medfilt2d
import matplotlib.pyplot as plt

from utility.common import *
from configs.modelConfigs import *
from configs.configs import SHOW


def selectClique(cliques):
    lens = [len(c) for c in cliques]
    groupCounts = [len(filteredCliqueEnds(c, gap=10)[0]) for c in cliques]
    ind = np.lexsort((lens, groupCounts))
    return cliques[ind[-1]]


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


def cliquesFromSSM(ssm_f, thresh=SSM_LOG_THRESH, show=SHOW):
    ssm = ssm_f[1]
    size = ssm.shape[-1]

    threshSSM = np.zeros_like(ssm, dtype=np.uint8)
    for i, j in product(range(size), range(size)):
        threshSSM[i, j] = 1 if ssm[i, j] >= thresh else 0

    g = nx.from_numpy_matrix(threshSSM)
    gShow = deepcopy(g)
    cliques = []
    while g:
        candidates = list(
            [c for i, c in zip(range(CLIQUE_CANDIDATES_COUNT), nx.find_cliques(g))]
        )
        c = selectClique(candidates)

        g.remove_nodes_from(c)
        cliques.insert(0, sorted(list(c)))
        if show:
            # ebunch = [(x, y) for x in c for y in range(size)]
            # gShow.remove_edges_from(ebunch)
            if len(cliques) % 10 == 0:
                mat = nx.to_numpy_matrix(gShow) * 5
                mat += getLabeledSSM(cliques, size)
                plt.imshow(mat)
                plt.show()
    cliques = sorted(cliques, key=lambda c: c[0])
    return cliques


def isAdjacent(cliqueA, cliqueB, dis=ADJACENT_DELTA_DISTANCE, dblock=0):
    _, x = filteredCliqueEnds(cliqueA)
    y, _ = filteredCliqueEnds(cliqueB)
    deltaX2y = np.array([np.min(np.abs(y - i)) for i in x])
    deltaY2x = np.array([np.min(np.abs(x - i)) for i in y])

    neighborX = np.sum(deltaX2y <= dis)
    neighborY = np.sum(deltaY2x <= dis)
    res = all([len(y) - neighborY <= dblock, len(x) - neighborX <= dblock,])
    # print(f"delta:{delta} res:{res} x,y:{x,y}")
    return res


def error(origCliques, mergedCliques, size, times, show=SHOW):
    olssm = getLabeledSSM(origCliques, size)
    mlssm = getLabeledSSM(mergedCliques, size)
    olssm[olssm > 0] = 1
    mlssm[mlssm > 0] = 1
    # false negative + false positive
    fnerr = np.sum((mlssm == 0) & olssm) / (np.sum(olssm) + EPSILON)
    fperr = np.sum(mlssm & (olssm == 0)) / (np.sum(olssm == 0) + EPSILON)
    fperr = max(0, fperr - 0.2)
    err = fnerr + fperr
    if show:
        print(f"errs:{fnerr:.5f},{fperr:.5f} sum:{err:.3f} len:{len(mergedCliques)}")
        x, xm = getLabeledSSM(origCliques, size), getLabeledSSM(mergedCliques, size)
        labels = [x[i, i] for i in range(size)]
        xm[xm > 0] = 10
        plt.imshow(x + xm)
        plt.plot(labels)
        plt.show()
    return err


def mergeAdjacentCliques(cliques, dis=ADJACENT_DELTA_DISTANCE, dblock=0):
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
        for dblock in range(2)
    ]
    # mclen = len(mergedCliquesList)
    # for i in range(1):
    #     mergedCliquesList.extend(
    #         [mergeAdjacentCliques(cs) for cs in mergedCliquesList[-mclen:]]
    #     )
    errors = [error(cliques, ncs, size, times) for ncs in mergedCliquesList]
    indices = np.argsort(errors)
    for i in indices:
        newCliques = mergedCliquesList[i]
        predicate = all(
            [
                len(newCliques) >= MIN_STRUCTURE_COUNT,
                # len(newCliques) <= MAX_STRUCTURE_COUNT,
            ]
        )
        if predicate:
            return newCliques
    print(
        f"[WARNING]:seqrecur failed, cliqueLengths={[len(x) for x in mergedCliquesList]}"
    )
    return mergedCliquesList[0]
