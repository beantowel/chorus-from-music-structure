import msaf
import os
import subprocess
from itertools import chain
from mir_eval.io import load_labeled_intervals

from models.classifier import *
from models.seqRecur import *
from models.pickSingle import *
from utility.dataset import *
from utility.common import *
from configs.modelConfigs import *
from configs.configs import ALGO_BASE_DIRS


def matchCliqueLabel(times, boundaries, cliques, dataset, idx):
    intervals = np.array(
        [(times[i], times[j]) for i, j in zip(boundaries[:-1], boundaries[1:])]
    )
    labels = np.full(intervals.shape[0], "others", dtype="U16")
    clabels = getCliqueLabels(dataset[idx]["gt"], cliques, intervals)
    for c, l in zip(cliques, clabels):
        for i in c:
            labels[i] = l
    mirexFmt = (intervals, labels)
    return mirexFmt


class AlgoSeqRecur:
    def __init__(self):
        self.clf = ChorusClassifier(CHORUS_CLASSIFIER_TRAIN_DATA_FILE["seqRecur"])

    def __call__(self, dataset, idx):
        ssm_f, mels_f = getFeatures(dataset, idx)
        cliques = self._process(dataset, idx, ssm_f)
        mirexFmt = chorusDetection(cliques, ssm_f[0], mels_f, self.clf)
        return mirexFmt

    def getStructure(self, dataset, idx):
        ssm_f, _ = getFeatures(dataset, idx)
        return self._process(dataset, idx, ssm_f)

    def _process(self, dataset, idx, ssm_f):
        tf = ExtractCliques()
        cliques_set = Preprocess_Dataset(tf.identifier, dataset, transform=tf.transform)
        cliquesSample = cliques_set[idx]
        origCliques = cliquesSample["cliques"]
        # origCliques = ssmStructure_sr(ssm_f)
        cliques = buildRecurrence(origCliques, ssm_f[0])
        return cliques


class AlgoSeqRecurSingle(AlgoSeqRecur):
    def __init__(self):
        super(AlgoSeqRecurSingle, self).__init__()

    def __call__(self, dataset, idx):
        mirexFmt = super(AlgoSeqRecurSingle, self).__call__(dataset, idx)
        _, mels_f = getFeatures(dataset, idx)
        mirexFmtSingle = maxArousal(mirexFmt, mels_f)
        return mirexFmtSingle


class AlgoSeqRecurBound:
    def __init__(self):
        self.rawAlgo = AlgoSeqRecur()

    def __call__(self, dataset, idx):
        ssm_f, _ = getFeatures(dataset, idx)
        cliques = self.rawAlgo._process(dataset, idx, ssm_f)

        boundaries = np.arange(ssm_f[0].shape[0], dtype=int)
        mirexFmt = matchCliqueLabel(ssm_f[0], boundaries, cliques, dataset, idx)
        return mirexFmt


# comment out the line 335 'file_struct.features_file = msaf.config.features_tmp_file'
# in XX/lib/python3.7/site-packages/msaf/run.py
# and 'mkdir features' in the dataset folder
# for faster performance using feature cache instead of single temporary feature
class MsafAlgos:
    def __init__(self, boundaries_id):
        # msaf.get_all_label_algorithms()：
        assert boundaries_id in ["vmo", "scluster", "cnmf"]
        self.bd = boundaries_id
        self.clf = ChorusClassifier(CHORUS_CLASSIFIER_TRAIN_DATA_FILE[boundaries_id])

    def __call__(self, dataset, idx):
        ssm_f, mels_f = getFeatures(dataset, idx)
        sample = dataset[idx]
        wavPath = sample["wavPath"]
        cliques = self._process(wavPath, ssm_f[0])
        mirexFmt = chorusDetection(cliques, ssm_f[0], mels_f, self.clf)
        return mirexFmt

    def getStructure(self, dataset, idx):
        ssm_f, _ = getFeatures(dataset, idx)
        sample = dataset[idx]
        wavPath = sample["wavPath"]
        return self._process(wavPath, ssm_f[0])

    def _process(self, wavPath, times):
        boundaries, labels = msaf.process(wavPath, boundaries_id=self.bd)
        tIntvs = np.array([boundaries[:-1], boundaries[1:]]).T
        arr = np.zeros(len(times) - 1, dtype=int)
        for tIntv, label in zip(tIntvs, labels):
            lower = np.searchsorted(times, tIntv[0])
            higher = np.searchsorted(times, tIntv[1])
            arr[lower:higher] = label
        cliques = cliquesFromArr(arr)
        newCliques = smoothCliques(cliques, len(times) - 1, SMOOTH_KERNEL_SIZE)
        return newCliques


class MsafAlgosBound:
    def __init__(self, boundaries_id):
        # msaf.get_all_boundary_algorithms():
        assert boundaries_id in ["scluster", "sf", "olda", "cnmf", "foote"]
        self.bd = boundaries_id

    def __call__(self, dataset, idx):
        sample = dataset[idx]
        wavPath = sample["wavPath"]
        gt = sample["gt"]
        boundaries, _ = msaf.process(wavPath, boundaries_id=self.bd)
        est_intvs = np.array([boundaries[:-1], boundaries[1:]]).T
        est_labels = matchLabel(est_intvs, gt)
        dur = librosa.get_duration(filename=wavPath)
        while est_intvs[-1][0] >= dur:
            est_intvs = est_intvs[:-1]
            est_labels = est_labels[:-1]
        est_intvs[-1][1] = dur
        return (est_intvs, est_labels)


class MsafAlgosBdryOnly:
    def __init__(self, boundaries_id):
        # msaf.get_all_label_algorithms()：
        assert boundaries_id in ["sf", "olda", "foote"]
        self.bd = boundaries_id
        self.clf = ChorusClassifier(CHORUS_CLASSIFIER_TRAIN_DATA_FILE[boundaries_id])

    def __call__(self, dataset, idx):
        ssm_f, mels_f = getFeatures(dataset, idx)
        sample = dataset[idx]
        wavPath = sample["wavPath"]
        cliques = self._process(wavPath, ssm_f)
        mirexFmt = chorusDetection(cliques, ssm_f[0], mels_f, self.clf)
        return mirexFmt

    def getStructure(self, dataset, idx):
        ssm_f, _ = getFeatures(dataset, idx)
        sample = dataset[idx]
        wavPath = sample["wavPath"]
        return self._process(wavPath, ssm_f)

    def _process(self, wavPath, ssm_f):
        times = ssm_f[0]
        boundaries, _ = msaf.process(wavPath, boundaries_id=self.bd)
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


class GroudTruthStructure:
    def __init__(self):
        self.clf = ChorusClassifier(CHORUS_CLASSIFIER_TRAIN_DATA_FILE["gtBoundary"])

    def getStructure(self, dataset, idx):
        tf = GenerateSSM()
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


class PopMusicHighlighter:
    def __init__(self, basedir=ALGO_BASE_DIRS["PopMusicHighlighter"]):
        self.basedir = basedir

    def __call__(self, dataset, idx):
        wavPath = dataset[idx]["wavPath"]
        title = dataset[idx]["title"]
        dur = librosa.get_duration(filename=wavPath)
        target = os.path.join(self.basedir, f"{title}_highlight.npy")
        assert os.path.exists(target)
        x = np.load(target)
        intervals = np.array([(0, x[0]), (x[0], x[1]), (x[1], dur),])
        labels = np.array(["others", "chorus", "others",], dtype="U16")
        return (intervals, labels)
