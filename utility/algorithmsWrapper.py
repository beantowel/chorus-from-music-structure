import os
import json
import subprocess
import librosa
import numpy as np
from itertools import chain
from pychorus import find_and_output_chorus
from mir_eval.io import load_labeled_intervals

from models.classifier import ChorusClassifier, chorusDetection, getFeatures
from utility.transform import ExtractCliques, GenerateSSM
from third_party.msaf.msafWrapper import process
from models.seqRecur import buildRecurrence, smoothCliques
from models.pickSingle import maxOverlap, tuneIntervals
from utility.dataset import DATASET_BASE_DIRS, Preprocess_Dataset, convertFileName
from utility.common import (
    cliquesFromArr,
    matchCliqueLabel,
    matchLabel,
    singleChorusSection,
    removeNumber,
    mergeIntervals,
    intervalIntersection,
)
from configs.modelConfigs import (
    CHORUS_DURATION,
    CHORUS_DURATION_SINGLE,
    SMOOTH_KERNEL_SIZE,
    SSM_LOG_THRESH,
    TUNE_WINDOW,
    CLF_TARGET_LABEL,
)
from configs.configs import logger, ALGO_BASE_DIRS


class AlgoSeqRecur:
    def __init__(self, trainFile):
        self.clf = ChorusClassifier(trainFile)

    def __call__(self, dataset, idx):
        ssm_f, mels_f = getFeatures(dataset, idx)
        cliques = self._process(dataset, idx, ssm_f)
        mirexFmt = chorusDetection(cliques, ssm_f[0], mels_f, self.clf)
        mirexFmt = tuneIntervals(
            mirexFmt, mels_f, chorusDur=CHORUS_DURATION, window=TUNE_WINDOW
        )
        return mirexFmt

    def getStructure(self, dataset, idx):
        ssm_f, _ = getFeatures(dataset, idx)
        return self._process(dataset, idx, ssm_f)

    def _process(self, dataset, idx, ssm_f):
        tf = ExtractCliques(dataset=dataset)
        cliques_set = Preprocess_Dataset(tf.identifier, dataset, transform=tf.transform)
        cliquesSample = cliques_set[idx]
        origCliques = cliquesSample["cliques"]
        # origCliques = ssmStructure_sr(ssm_f)
        cliques = buildRecurrence(origCliques, ssm_f[0])
        return cliques


class AlgoSeqRecurSingle(AlgoSeqRecur):
    def __init__(self, trainFile):
        super(AlgoSeqRecurSingle, self).__init__(trainFile)

    def __call__(self, dataset, idx):
        ssm_f, mels_f = getFeatures(dataset, idx)
        cliques = self._process(dataset, idx, ssm_f)
        mirexFmt = chorusDetection(cliques, ssm_f[0], mels_f, self.clf)
        mirexFmtSingle = maxOverlap(
            mirexFmt, chorusDur=CHORUS_DURATION_SINGLE, centering=False
        )
        mirexFmtSingle = tuneIntervals(
            mirexFmtSingle, mels_f, chorusDur=CHORUS_DURATION_SINGLE, window=TUNE_WINDOW
        )
        return mirexFmtSingle


class AlgoSeqRecurBound:
    def __init__(self, trainFile):
        self.rawAlgo = AlgoSeqRecur(trainFile)

    def __call__(self, dataset, idx):
        ssm_f, mels_f = getFeatures(dataset, idx)
        cliques = self.rawAlgo._process(dataset, idx, ssm_f)

        times = ssm_f[0]
        intervals = np.array([(times[i], times[i + 1]) for i in range(len(times) - 1)])
        mirexFmt = matchCliqueLabel(intervals, cliques, dataset[idx]["gt"])
        mirexFmt = tuneIntervals(
            mirexFmt, mels_f, chorusDur=CHORUS_DURATION, window=TUNE_WINDOW
        )
        return mirexFmt


class BaseMsafAlgos:
    def __init__(self, boundaries_id, trainFile, valid_ids):
        # msaf.get_all_label_algorithms()：
        assert boundaries_id in valid_ids
        self.bd = boundaries_id
        self.clf = ChorusClassifier(trainFile)
        self.cacheDir = os.path.join(
            DATASET_BASE_DIRS["LocalTemporary_Dataset"], "msaf-cache"
        )

    def __call__(self, dataset, idx):
        ssm_f, mels_f = getFeatures(dataset, idx)
        cliques = self._process(dataset, idx, ssm_f)
        mirexFmt = chorusDetection(cliques, ssm_f[0], mels_f, self.clf)
        return mirexFmt

    def getStructure(self, dataset, idx):
        ssm_f, _ = getFeatures(dataset, idx)
        return self._process(dataset, idx, ssm_f)

    def cacheFile(self, dataset, idx):
        title = dataset[idx]["title"]
        dname = dataset.__class__.__name__
        feature_file = os.path.join(self.cacheDir, f"{dname}-{title}-feat.json")
        est_file = os.path.join(self.cacheDir, f"{dname}-{title}-est.jams")
        return feature_file, est_file

    def _process(self, dataset, idx, ssm_f):
        raise NotImplementedError


class MsafAlgos(BaseMsafAlgos):
    def __init__(self, boundaries_id, trainFile):
        super(MsafAlgos, self).__init__(
            boundaries_id, trainFile, ["vmo", "scluster", "cnmf"]
        )

    def _process(self, dataset, idx, ssm_f):
        wavPath = dataset[idx]["wavPath"]
        times = ssm_f[0]
        feat, est = self.cacheFile(dataset, idx)
        boundaries, labels = process(wavPath, self.bd, feat, est)
        tIntvs = np.array([boundaries[:-1], boundaries[1:]]).T
        arr = np.zeros(len(times) - 1, dtype=int)
        for tIntv, label in zip(tIntvs, labels):
            lower = np.searchsorted(times, tIntv[0])
            higher = np.searchsorted(times, tIntv[1])
            arr[lower:higher] = label
        cliques = cliquesFromArr(arr)
        newCliques = smoothCliques(cliques, len(times) - 1, SMOOTH_KERNEL_SIZE)
        return newCliques


class MsafAlgosBdryOnly(BaseMsafAlgos):
    def __init__(self, boundaries_id, trainFile):
        super(MsafAlgosBdryOnly, self).__init__(
            boundaries_id, trainFile, ["sf", "olda", "foote"]
        )

    def _process(self, dataset, idx, ssm_f):
        wavPath = dataset[idx]["wavPath"]
        feat, est = self.cacheFile(dataset, idx)
        boundaries, _ = process(wavPath, self.bd, feat, est)
        times = ssm_f[0]
        tIntvs = np.array([boundaries[:-1], boundaries[1:]]).T

        blockSSM = np.zeros((len(tIntvs), len(tIntvs)))
        for i, (xbegin, xend) in enumerate(tIntvs):
            # left	a[i-1] < v <= a[i]
            # right	a[i-1] <= v < a[i]
            xlower = np.searchsorted(times, xbegin)
            xhigher = np.searchsorted(times, xend)
            for j, (ybegin, yend) in enumerate(tIntvs):
                ylower = np.searchsorted(times, ybegin)
                yhigher = np.searchsorted(times, yend)
                size = (yhigher - ylower) * (xhigher - xlower)
                if size > 0:
                    blockSSM[i, j] = np.sum(
                        ssm_f[1][xlower:xhigher, ylower:yhigher] > SSM_LOG_THRESH
                    )
                    blockSSM[i, j] = blockSSM[i, j] / size
        labels = np.arange(len(tIntvs), dtype=int)
        for i in range(len(labels)):
            score = [blockSSM[i, j] for j in chain(range(i), range(i + 1, len(labels)))]
            labelIdx = np.argmax(score)
            if labelIdx < i:
                labels[i] = labelIdx

        arr = np.zeros(len(times) - 1, dtype=int)
        for tIntv, label in zip(tIntvs, labels):
            lower = np.searchsorted(times, tIntv[0])
            higher = np.searchsorted(times, tIntv[1])
            arr[lower:higher] = label
        cliques = cliquesFromArr(arr)
        newCliques = smoothCliques(cliques, len(times) - 1, SMOOTH_KERNEL_SIZE)
        return newCliques


class MsafAlgosBound(BaseMsafAlgos):
    def __init__(self, boundaries_id):
        super(MsafAlgosBound, self).__init__(
            boundaries_id, None, ["scluster", "sf", "olda", "cnmf", "foote"]
        )

    def __call__(self, dataset, idx):
        sample = dataset[idx]
        wavPath = sample["wavPath"]
        gt = sample["gt"]
        feat, est = self.cacheFile(dataset, idx)
        boundaries, _ = process(wavPath, self.bd, feat, est)
        est_intvs = np.array([boundaries[:-1], boundaries[1:]]).T
        est_labels = matchLabel(est_intvs, gt)
        dur = librosa.get_duration(filename=wavPath)
        while est_intvs[-1][0] >= dur:
            est_intvs = est_intvs[:-1]
            est_labels = est_labels[:-1]
        est_intvs[-1][1] = dur
        return (est_intvs, est_labels)


class GroudTruthStructure:
    def __init__(self, trainFile):
        self.clf = ChorusClassifier(trainFile)

    def getStructure(self, dataset, idx):
        tf = GenerateSSM(dataset=dataset)
        target = Preprocess_Dataset(tf.identifier, dataset, transform=tf.transform)[
            idx
        ]["target"]
        cliques = cliquesFromArr([target[i, i] for i in range(target.shape[0])])
        return cliques

    def __call__(self, dataset, idx):
        ssm_f, mels_f = getFeatures(dataset, idx)
        cliques = self.getStructure(dataset, idx)
        mirexFmt = chorusDetection(cliques, ssm_f[0], mels_f, self.clf)
        return mirexFmt


class CachedAlgo:
    def __init__(self, dirname, baseDir=DATASET_BASE_DIRS["LocalTemporary_Dataset"]):
        self.cacheDir = os.path.join(baseDir, dirname)
        if not os.path.exists(self.cacheDir):
            os.mkdir(self.cacheDir)

    def _cacheFile(self, dataset, idx):
        title = dataset[idx]["title"]
        return os.path.join(
            self.cacheDir, f"{dataset.__class__.__name__}-{idx}-{title}.json"
        )

    def readCache(self, dataset, idx):
        filename = self._cacheFile(dataset, idx)
        if not os.path.exists(filename):
            return None
        with open(filename) as f:
            data = json.load(f)
            return data

    def writeCache(self, dataset, idx, data):
        filename = self._cacheFile(dataset, idx)
        with open(filename, "w") as f:
            logger.info(f"writing to cache, path={filename}")
            json.dump(data, f)


class PopMusicHighlighter(CachedAlgo):
    def __init__(self):
        super(PopMusicHighlighter, self).__init__("highlighter-cache")
        self.algoDir = ALGO_BASE_DIRS["PopMusicHighlighter"]
        if not os.path.exists(os.path.join(self.algoDir, "venv")):
            ret = subprocess.call(
                "./init.sh", shell=True, cwd=self.algoDir, executable="/bin/bash"
            )
            assert ret == 0, f"return value: {ret} != 0"

    def getChorus(self, wavPath):
        title = os.path.splitext(os.path.basename(wavPath))[0]
        output = os.path.join(ALGO_BASE_DIRS["TmpDir"], f"{title}_highlighter_out.txt")
        commands = ("./venv/bin/python", "wrapper.py", wavPath, output)
        ret = subprocess.call(commands, cwd=self.algoDir)
        assert ret == 0, f"return value: {ret} != 0"
        intervals, labels = load_labeled_intervals(output, delimiter="\t")
        assert labels[1] == "chorus", f"can't find chorus, labels={labels}"
        return intervals[1][0], intervals[1][1]

    def __call__(self, dataset, idx):
        wavPath = dataset[idx]["wavPath"]
        dur = librosa.get_duration(filename=wavPath)
        data = self.readCache(dataset, idx)
        if data is not None:
            start, end = data["start"], data["end"]
        else:
            start, end = self.getChorus(wavPath)
            self.writeCache(dataset, idx, {"start": start, "end": end})
        return singleChorusSection(start, end, dur)


class RefraiD(CachedAlgo):
    def __init__(self):
        super(RefraiD, self).__init__("RefraiD-cache")

    def getChorus(self, wavPath, clip_length=30):
        start = find_and_output_chorus(wavPath, None, clip_length)
        while start is None and clip_length > 5:
            clip_length -= 5
            logger.warn(
                f"RefraiD failed to detect chorus, reduce clip_length={clip_length}"
            )
            start = find_and_output_chorus(wavPath, None, clip_length)
        if start is None:
            logger.warn(f"RefraiD failed to detect chorus")
            start = 0
        return start, clip_length

    def __call__(self, dataset, idx):
        wavPath = dataset[idx]["wavPath"]
        dur = librosa.get_duration(filename=wavPath)
        data = self.readCache(dataset, idx)
        if data is not None:
            start, length = data["start"], data["length"]
        else:
            start, length = self.getChorus(wavPath)
            self.writeCache(dataset, idx, {"start": start, "length": length})
        return singleChorusSection(start, start + length, dur)


class AlgoMixed:
    def __init__(self, trainFile):
        self.pred1 = AlgoSeqRecur(trainFile=trainFile)
        self.clf = self.pred1.clf
        self.pred2 = PopMusicHighlighter()

    def mixChorus(self, mirex1, mirex2):
        mirex1, mirex2 = removeNumber(mirex1), removeNumber(mirex2)
        mirex1, mirex2 = mergeIntervals(mirex1), mergeIntervals(mirex2)
        chorus1 = np.nonzero(np.char.startswith(mirex1[1], CLF_TARGET_LABEL))[0]
        chorus2 = np.nonzero(np.char.startswith(mirex2[1], CLF_TARGET_LABEL))[0]
        logger.debug(f"choru1={chorus1} chorus2={chorus2}")
        dur = mirex1[0][-1][1]

        chorusIntsec = (
            []
        )  # select (begin, begin + 30s) with maximal overlap with detected chorus sections
        for idx1 in chorus1:
            intsec = np.sum(
                [
                    intervalIntersection(mirex1[0][idx1], mirex2[0][idx2])
                    for idx2 in chorus2
                ]
            )
            chorusIntsec.append(intsec)
        nonzeros = np.nonzero(chorusIntsec)[0]
        idx = chorus1[nonzeros[0]] if len(nonzeros) > 0 else 0

        begin, end = mirex1[0][idx]
        return singleChorusSection(begin, end, dur)

    def __call__(self, dataset, idx):
        out1 = self.pred1(dataset, idx)
        out2 = self.pred2(dataset, idx)
        mixed = self.mixChorus(out1, out2)
        return mixed
