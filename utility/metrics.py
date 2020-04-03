import os
import pickle
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from mir_eval import segment, transcription
from multiprocessing import Pool

from tqdm import tqdm

from utility.common import *
from configs.configs import METRIC_NAMES, PLOT_METRIC_FIELDS, DETECTION_WINDOW, NUM_WORKERS


class AlgoEvaluator():
    def __init__(self, dataset, algo, num_workers=NUM_WORKERS):
        self.dataset = dataset
        self.algo = algo
        self.num_workers = num_workers

    def eval(self, idx):
        est = self.algo(self.dataset, idx)
        sample = self.dataset[idx]
        # print(f'title[{idx}]:{sample["title"]}')
        gt = sample['gt']
        metric = getMetric(gt, est)
        return metric

    def __call__(self):
        try:
            with Pool(self.num_workers) as p:
                N = len(self.dataset)
                metrics = list(tqdm(p.imap(self.eval, range(N)), total=N))
        except RuntimeError as e:
            # CUDA RuntimeError
            metrics = [self.eval(i) for i in tqdm(range(N))]
            print('[RuntimeError]', e)
        titles = [sample['title'] for sample in self.dataset]
        return np.array(metrics), titles


def chorusOverlap(ref, est, beta=1.0):
    ref, est = deepcopy(ref), deepcopy(est)
    ref_intvs, est_intvs = filterIntvs(ref), filterIntvs(est)
    totalItsc = 0
    for est_intv in est_intvs:
        itsc = sum([intervalIntersection(est_intv, ref_intv)
                    for ref_intv in ref_intvs])
        totalItsc += itsc
    ref_duration = np.sum([intv[1] - intv[0] for intv in ref_intvs])
    est_duration = np.sum([intv[1] - intv[0] for intv in est_intvs])
    P = 0 if est_duration == 0 else totalItsc / est_duration
    assert ref_duration > 0
    R = totalItsc / ref_duration
    F = 0 if P == 0 and R == 0 else (1 + beta**2) * P * R / (R + P * beta**2)
    return P, R, F


def chorusOverlapNear(ref, est, beta=1.0):
    def distance(intv0, intv1):
        mid0 = (intv0[0] + intv0[1]) / 2
        mid1 = (intv1[0] + intv1[1]) / 2
        return abs(mid0 - mid1)

    ref, est = deepcopy(ref), deepcopy(est)
    ref = (ref[0], extractFunctions(ref[1]))
    est = (est[0], extractFunctions(est[1]))
    ref, est = mergeIntervals(ref), mergeIntervals(est)
    ref_intvs, est_intvs = filterIntvs(ref), filterIntvs(est)

    totalItsc = 0
    nearest_refs = []
    for est_intv in est_intvs:
        dis = [distance(ref_intv, est_intv) for ref_intv in ref_intvs]
        idx = np.argmin(dis)
        nearest_refs.append(idx)
        totalItsc += intervalIntersection(est_intv, ref_intvs[idx])
    nearest_refs = list(set(nearest_refs))
    est_duration = np.sum([intv[1] - intv[0] for intv in est_intvs])
    ref_duration = np.sum([ref_intvs[i][1] - ref_intvs[i][0]
                           for i in nearest_refs])
    P = 0 if est_duration == 0 else totalItsc / est_duration
    R = 0 if ref_duration == 0 else totalItsc / ref_duration
    F = 0 if P == 0 and R == 0 else (1 + beta**2) * P * R / (R + P * beta**2)
    return P, R, F


def chorusOnsetPRF(ref, est):
    ref, est = deepcopy(ref), deepcopy(est)
    ref = (ref[0], extractFunctions(ref[1], ['chorus', 'verse']))
    est = (est[0], extractFunctions(est[1], ['chorus', 'verse']))
    ref, est = mergeIntervals(ref), mergeIntervals(est)
    ref_intvs, est_intvs = filterIntvs(ref), filterIntvs(est)

    if len(est_intvs) == 0 and len(ref_intvs) != 0:
        dtct = 0, 0, 0
    elif len(ref_intvs) == 0:
        dtct = 1, 1, 1
    else:
        dtct = transcription.onset_precision_recall_f1(
            ref_intvs, est_intvs, onset_tolerance=DETECTION_WINDOW)
    return dtct


def getMetric(ref, est):
    ovlp = chorusOverlap(ref, est)
    sovl = chorusOverlapNear(ref, est)
    dtct = chorusOnsetPRF(ref, est)
    ovlp_P, ovlp_R, ovlp_F = ovlp
    sovl_P, sovl_R, sovl_F = sovl
    dtct_P, dtct_R, dtct_F = dtct
    return ovlp_P, ovlp_R, ovlp_F, sovl_P, sovl_R, sovl_F, dtct_P, dtct_R, dtct_F


class Metrics_Saver():
    def __init__(self, datasetName):
        self.datasetName = datasetName
        self.algoNames = []
        self.metricsList = []  # (algorithms, songs, metricFields)
        self.titlesList = []

    def addResult(self, algoName, metrics, titles):
        self.algoNames.append(algoName)
        self.metricsList.append(metrics)
        self.titlesList.append(titles)

    def removeResult(self, algoName):
        try:
            # remove all match algoName result
            while True:
                idx = self.algoNames.index(algoName)
                self.algoNames.pop(idx)
                self.metricsList.pop(idx)
                self.titlesList.pop(idx)
        except ValueError:
            print(f'all {algoName} result removed')

    def reWriteResult(self, algoName, metrics, titles):
        assert algoName in self.algoNames, f'{algoName} not in {self.algoNames}'
        idx = self.algoNames.index(algoName)
        self.metricsList[idx] = metrics
        self.titlesList[idx] = titles

    def getResult(self, algoName, titles=None):
        try:
            idx = self.algoNames.index(algoName)
            metrics = self.metricsList[idx]
            _titles = self.titlesList[idx]
            titles = _titles if titles is None else titles
            res = [metrics[_titles.index(title)] for title in titles]
            return np.array(res), titles
        except ValueError:
            print(f'{algoName} results not found in {self.datasetName}')

    def writeFullResults(self, dirname):
        fullOutputFile = os.path.join(dirname, f'{self.datasetName}_full.csv')
        cols = ['title', 'algo'] + METRIC_NAMES
        df = pd.DataFrame(columns=cols)
        for algoName, metrics, titles in zip(self.algoNames, self.metricsList, self.titlesList):
            n = len(titles)
            head = np.array([titles, [algoName] * n]).T
            headDf = pd.DataFrame(data=head, columns=cols[:2])
            metricDf = pd.DataFrame(data=metrics, columns=cols[2:])
            algoDf = pd.concat([headDf, metricDf], axis=1)
            df = pd.concat([df, algoDf], ignore_index=True)
        df.to_csv(fullOutputFile)
        print(f'results written to {fullOutputFile}')

    def writeAveResults(self, dirname):
        aveOutputFile = os.path.join(dirname, f'{self.datasetName}.csv')
        columns = ['algo'] + METRIC_NAMES
        df = pd.DataFrame(columns=columns)
        for algoName, metrics in zip(self.algoNames, self.metricsList):
            data = np.hstack(
                [algoName, np.mean(metrics, axis=0)]).reshape(1, -1)
            df = pd.concat([df, pd.DataFrame(data=data, columns=columns)])
        df.to_csv(aveOutputFile)
        print(f'results written to {aveOutputFile}')

    def saveViolinPlot(self, dirname, plotMetric=PLOT_METRIC_FIELDS, order=None):
        matplotlib.use('Agg')
        pltOutputFile = os.path.join(dirname, f'{self.datasetName}.svg')
        rows, cols = len(plotMetric), len(plotMetric[0])
        axisNames = np.array(plotMetric).flatten()
        metricsList = np.array(self.metricsList)
        if order is not None:
            algoNames = list(filter(lambda x: x in self.algoNames, order))
            metricsList = metricsList[
                [self.algoNames.index(aName) for aName in algoNames],
                :,
                :]
        else:
            algoNames = self.algoNames
        metricsFieldSelector = np.array(
            [METRIC_NAMES.index(name) for name in axisNames])
        metricsList = metricsList[:, :, metricsFieldSelector]

        pos = np.arange(len(algoNames), dtype=int) + 1
        fig, axes = plt.subplots(
            nrows=rows, ncols=cols, figsize=(cols*4*len(algoNames)/10, rows*4))
        for i, axis in enumerate(axes.flatten()):
            data = [metrics[:, i] for metrics in metricsList]
            axis.violinplot(data, pos, showmeans=True, showextrema=True)
            axis.set_title(axisNames[i])
            plt.setp(axis.get_xticklabels(), rotation=45)

        # plt.suptitle(self.datasetName, fontsize=20)
        plt.setp(axes, xticks=pos, xticklabels=algoNames)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(pltOutputFile, quality=100)
        print(f'violin plot written to {pltOutputFile}')

    def dump(self, dirname):
        dumpFile = os.path.join(dirname, f'{self.datasetName}.pkl')
        with open(dumpFile, 'wb') as f:
            pickle.dump((self.datasetName, self.algoNames,
                         self.metricsList, self.titlesList), f, pickle.HIGHEST_PROTOCOL)
        print(f'saver object written to {dumpFile}')
        return self

    def load(self, dirname):
        dumpFile = os.path.join(dirname, f'{self.datasetName}.pkl')
        try:
            with open(dumpFile, 'rb') as f:
                dname, self.algoNames, self.metricsList, self.titlesList = pickle.load(
                    f)
                if dname != self.datasetName:
                    print(
                        f'[WARNING]:old name:<{dname}> != new name:<{self.datasetName}>')
            print(f'saver object loaded from {dumpFile}')
        except FileNotFoundError:
            print(f'saver object {dumpFile} not found, set to empty')
        return self
