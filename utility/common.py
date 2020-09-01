import numpy as np
from typing import List
import matplotlib.pyplot as plt

from configs.configs import DEBUG, logger
from configs.modelConfigs import *
from collections import defaultdict


def cliqueTails(clique):
    nxt = np.array(clique) + 1
    nxt = sorted(set(nxt) - set(clique))
    return nxt


def cliqueHeads(clique):
    prv = np.array(clique) - 1
    prv = sorted(set(prv) - set(clique))
    prv = [x + 1 for x in prv]
    return prv


def cliqueGroups(clique):
    heads = cliqueHeads(clique)
    groups = [[clique[0]]]
    for idx in clique[1:]:
        if idx in heads:
            groups.append([idx])
        else:
            groups[-1].append(idx)
    return groups


def filteredCliqueEnds(clique, min_size=1, gap=5):
    groups = cliqueGroups(clique)
    heads = [group[0] for group in groups if len(group) >= min_size]
    tails = [group[-1] + 1 for group in groups if len(group) >= min_size]
    if len(heads) > 0:
        hs, ts = [heads[0]], []
        for nxtHead, tail in zip(heads[1:], tails[:-1]):
            if nxtHead - tail >= gap:
                hs.append(nxtHead)
                ts.append(tail)
        ts.append(tails[-1])
        return np.array(hs), np.array(ts)
    else:
        return np.array([]), np.array([])


def intervalIntersection(intv0, intv1):
    x = min(intv0[1], intv1[1]) - max(intv0[0], intv1[0])
    x = 0 if x < 0 else x
    return x


def filterIntvs(mirexFmt, fun="chorus"):
    intvs, labels = mirexFmt
    labels = extractFunctions(labels, [fun])
    intvs = intvs[labels == fun]
    return intvs


def mergeIntervals(mirexFmt):
    intervals, labels = mirexFmt
    new_intervals = [intervals[0]]
    new_labels = [labels[0]]
    for interval, label in zip(intervals[1:], labels[1:]):
        if label == new_labels[-1]:
            new_intervals[-1][1] = interval[1]
        else:
            new_intervals.append(interval)
            new_labels.append(label)
    new_intervals = np.array(new_intervals)
    new_labels = np.array(new_labels, dtype="U16")
    return (new_intervals, new_labels)


def extractFunctions(labels: np.ndarray, funs: List[str] = ["chorus"]) -> np.ndarray:
    newLabels = []
    for lab in labels:
        preds = list(map(lambda fun: lab.lower().startswith(fun), funs))
        if any(preds):
            newLabels.append(funs[preds.index(True)])
        else:
            newLabels.append("others")
    return np.array(newLabels, dtype="U16")


def matchLabel(est_intvs, gt):
    ref_intvs = filterIntvs(gt)
    gt_est_labels = []
    for onset, offset in est_intvs:
        intersec = sum(
            [intervalIntersection(intv, (onset, offset)) for intv in ref_intvs]
        )
        est_dur = offset - onset
        predicate = all(
            [
                intersec >= est_dur / 2,
            ]
        )
        label = "chorus" if predicate else "others"
        gt_est_labels.append(label)
    return np.array(gt_est_labels)


def matchCliqueLabel(intervals, cliques, gt):
    labels = np.full(intervals.shape[0], "others", dtype="U16")
    clabels = getCliqueLabels(gt, cliques, intervals)
    for c, l in zip(cliques, clabels):
        for i in c:
            labels[i] = l
    mirexFmt = (intervals, labels)
    return mirexFmt


def getCliqueLabels(gt, cliques, intervals):
    gt = mergeIntervals(gt)
    ref_intvs = filterIntvs(gt)
    cliqueLabels = []
    for clique in cliques:
        cintvs = [
            (intervals[group[0]][0], intervals[group[-1]][1])
            for group in cliqueGroups(clique)
        ]
        intersec = np.sum(
            [
                intervalIntersection(intv, cintv)
                for cintv in cintvs
                for intv in ref_intvs
            ]
        )
        cdur = sum([(offset - onset) for onset, offset in cintvs])
        hit_ref_duration = np.sum(
            [
                intv[1] - intv[0]
                for intv in ref_intvs
                if np.sum([intervalIntersection(intv, cintv) for cintv in cintvs]) > 0
            ]
        )
        p = intersec / cdur if cdur > 0 else 0
        r = intersec / hit_ref_duration if hit_ref_duration > 0 else 0
        predicate = all(
            [
                p >= CC_PRECISION,
                r >= CC_RECALL,
                cdur >= MINIMUM_CHORUS_DUR,
            ]
        )
        # ml = matchLabel(cintvs, gt)
        # predicate = sum(ml == 'chorus') >= len(ml) * 0.5
        label = "chorus" if predicate else "others"
        cliqueLabels.append(label)
    return cliqueLabels


def cliquesFromArr(arr):
    # key:clique label
    # value:frame number list
    cliquesDic = defaultdict(list)
    for i, label in enumerate(arr):
        cliquesDic[label].append(i)
    newCliques = list(cliquesDic.values())
    newCliques = sorted(newCliques, key=lambda c: c[0])
    return newCliques


def getLabeledSSM(cliques, size):
    boundaries = np.arange(size + 1, dtype=int)
    labeledSSM = np.zeros((size, size), dtype=int)
    for flag, clique in enumerate(cliques):
        groups = cliqueGroups(clique)
        for xgrp in groups:
            for ygrp in groups:
                xbegin, xend = boundaries[xgrp[0]], boundaries[xgrp[-1] + 1]
                ybegin, yend = boundaries[ygrp[0]], boundaries[ygrp[-1] + 1]
                labeledSSM[xbegin:xend, ybegin:yend] = flag + 1
    return labeledSSM


def logSSM(ssm, inplace=True):
    if not inplace:
        ssm = ssm.copy()
    ssm[ssm < 0] = 0
    ssm += EPSILON
    ssm = np.log(ssm)
    return ssm


def expSSM(ssm, inplace=True):
    if not inplace:
        ssm = ssm.copy()
    ssm = np.exp(ssm)
    ssm -= EPSILON
    ssm[ssm < 0] = 0
    return ssm


def singleChorusSection(begin, end, dur):
    intervals = np.array([(0, begin), (begin, end), (end, dur)])
    labels = np.array(
        ["others", "chorus", "others"],
        dtype="U16",
    )
    return (intervals, labels)


def printArray(arr, name, show=False):
    logger.debug(f"{name}{arr.shape}, min={np.min(arr)} max={np.max(arr)}")
    if show:
        plt.imshow(logSSM(arr), aspect="auto")
        plt.colorbar()
        plt.show()
