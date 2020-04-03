import numpy as np
import networkx as nx
from copy import copy, deepcopy
from itertools import product
from collections import defaultdict
from queue import Queue
from scipy.stats import mode
import matplotlib.pyplot as plt

from utility.GraphDitty.CSMSSMTools import getCSMCosine
from utility.common import *
from configs.modelConfigs import *


def ssmStructure_sr(ssm_f, thresh=SSM_LOG_THRESH):
    ssm = ssm_f[1]

    size = ssm.shape[-1]
    threshSSM = np.zeros_like(ssm, dtype=int)
    for i, j in product(range(size), range(size)):
        threshSSM[i, j] = 1 if ssm[i, j] >= thresh else 0

    g = nx.from_numpy_matrix(threshSSM)
    cliques = []
    while g:
        candidates = list(
            [c for i, c in zip(range(CLIQUE_CANDIDATES_COUNT), nx.find_cliques(g))])
        scores = [len(c) for c in candidates]
        c = candidates[np.argmax(scores)]
        g.remove_nodes_from(c)
        cliques.insert(0, sorted(list(c)))
    cliques = sorted(cliques, key=lambda c: c[0])
    return cliques


def isAdjacent(cliqueA, cliqueB, dis=ADJACENT_DELTA_DISTANCE, dblock=1):
    _, x = filteredCliqueEnds(cliqueA)
    y, _ = filteredCliqueEnds(cliqueB)
    dsize = max(len(x), len(y)) - dblock
    if len(x) <= len(y):
        delta = np.array([np.min(np.abs(y - i)) for i in x])
    else:
        delta = np.array([np.min(np.abs(x - i)) for i in y])
    res = all([
        np.sum(delta <= dis) >= max(dsize, 1),
        np.abs(len(x) - len(y)) <= dblock,
    ])
    # if res:
    #     print(f'delta:{delta} res:{res} x,y:{x,y}')
    return res


def error(origCliques, mergedCliques, size, times):
    olssm = getLabeledSSM(origCliques, size)
    mlssm = getLabeledSSM(mergedCliques, size)
    olssm[olssm > 0] = 1
    mlssm[mlssm > 0] = 1
    # false negative + false positive
    fnerr = np.sum((mlssm == 0) & olssm) / (np.sum(olssm) + EPSILON)
    fperr = np.sum(mlssm & (olssm == 0)) / (np.sum(olssm == 0) + EPSILON)
    fperr = max(0, fperr - 0.1)
    err = fnerr + fperr
    if [True, False][1]:
        print(
            f'errs:{fnerr:.3f},{fperr:.3f} sum:{err:.3f} len:{len(mergedCliques)}')
        x = getLabeledSSM(origCliques, size)
        plt.imshow(getLabeledSSM(mergedCliques, size) + x)
        plt.show()
    return err


def mergeAdjacentCliques(cliques, dblock):
    size = len(cliques)
    adjMat = np.zeros([size]*2, dtype=int)
    # calculate adjacency matrix
    for i in range(size):
        adjMat[i, i] = 1
        for j in range(i+1, size):
            res = isAdjacent(cliques[i], cliques[j], dblock=dblock)
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


def smoothCliques(cliques, size, kernel_size=SMOOTH_KERNEL_SIZE):
    # arr[<frames>]:<label>
    arr = np.zeros(size, dtype=int)
    for i, clique in enumerate(cliques):
        for idx in clique:
            arr[idx] = i
    dt = (kernel_size-1) // 2
    while True:
        newarr = np.zeros_like(arr, dtype=int)
        for i in range(len(arr)):
            window = arr[max(0, i-dt):min(len(arr), i+dt+1)]
            m, _ = mode(window)
            newarr[i] = m
        if (arr == newarr).all():
            break
        else:
            arr = deepcopy(newarr)

    newCliques = cliquesFromArr(arr)
    return newCliques


def buildRecurrence(cliques, times, dblockRng=DELTA_BLOCK_RANGE):
    cliques = deepcopy(cliques)
    size = len(times) - 1
    newCliquesList = [
        smoothCliques(
            mergeAdjacentCliques(cliques, dblock),
            size,
            kernel_size=ksize
        ) for dblock in range(dblockRng) for ksize in SMOOTH_KERNEL_RANGE
    ]
    newCliquesList.extend([
        smoothCliques(
            mergeAdjacentCliques(cs, 0),
            size
        ) for cs in newCliquesList])
    errors = [error(cliques, ncs, size, times) for ncs in newCliquesList]
    indices = np.argsort(errors)
    for idx in indices:
        newCliques = newCliquesList[idx]
        predicate = all([
            len(newCliques) >= MIN_STRUCTURE_COUNT,
            # len(newCliques) <= MAX_STRUCTURE_COUNT,
        ])
        if predicate:
            return newCliques
    print(f'[WARNING]:seqrecur failed:{[len(x) for x in newCliquesList]}')
    return newCliquesList[0]
