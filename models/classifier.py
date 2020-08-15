import librosa
import pickle
import numpy as np
from scipy import signal
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from models.seqRecur import *
from utility.common import *
from utility.transform import *
from configs.modelConfigs import *


def chorusDetection(cliques, ssm_times, mels_f, clf):
    boundaries = np.arange(len(ssm_times))
    features = getCliqueFeatures(cliques, boundaries, ssm_times, mels_f)
    cindices = clf.predict(features)
    indices = [i for cidx in cindices for i in cliques[cidx]]

    # assign labels (maximum length=16)
    labels = np.full(len(boundaries) - 1, "others", dtype="U16")
    labels[indices] = "chorus"
    intervals = np.array(
        [(ssm_times[i], ssm_times[j]) for i, j in zip(boundaries[:-1], boundaries[1:])]
    )
    labels = numberCliques(cliques, labels)
    # frame based intervals and labels
    # merge the output using `mergeIntervals` for segment based usage
    mirexFmt = (intervals, labels)
    return mirexFmt


def numberCliques(cliques, labels):
    # numbering cliques (recurrence label)
    typeCount = {}
    for clique in sorted(cliques, key=lambda c: c[0]):
        ltype = labels[clique[0]]
        count = typeCount.get(ltype, 0)
        for idx in clique:
            labels[idx] += f" {chr(65+count)}"
        typeCount[ltype] = count + 1
    return labels


def sliceTimeSeries(times, values, tIntvs):
    res = []
    for tIntv in tIntvs:
        lower = np.searchsorted(times, tIntv[0])
        higher = np.searchsorted(times, tIntv[1])
        res.extend(values[lower:higher])
    res = [0] if len(res) == 0 else res
    return np.array(res)


def getCliqueFeatures(cliques, boundaries, ssm_times, mels_f):
    def time(idx):
        return ssm_times[boundaries[idx]]

    def getDuration(clique):
        dur = sum([time(idx + 1) - time(idx) for idx in clique])
        return dur

    def getCount(clique, min_gap=15):
        tails = cliqueTails(clique)
        heads = cliqueHeads(clique)
        flt = [
            time(head) - time(tail)
            for head, tail in zip(heads[1:], tails[:-1])
            if time(head) - time(tail) > min_gap
        ]
        return len(flt) + 1

    def getAudioFeature(clique):
        tIntvs = [
            (time(h), time(t)) for h, t in zip(cliqueHeads(clique), cliqueTails(clique))
        ]

        mels = sliceTimeSeries(mels_f[0], mels_f[1], tIntvs)
        voicingRate = np.sum(mels > 0) / len(mels)
        mels = mels[mels > 0]
        mels = mels if len(mels) > 0 else np.array([0])
        melMedian = np.median(mels)
        melMin = np.min(mels)
        melMax = np.max(mels)
        dur = ssm_times[-1]
        count = getCount(clique)
        cdur = getDuration(clique) / dur
        heads = cliqueHeads(clique)
        head = time(heads[0]) / dur
        headx = time(heads[-1]) / dur

        feature = cdur, voicingRate, melMedian, melMin, melMax, head, headx, count
        return feature

    def getRelativeFeature(features):
        relSelector = [0, 1, 2, 3, 4, 5, 6, 7]
        relmaxs = np.max(features[:, relSelector], axis=0)
        rels = features[:, relSelector] / (relmaxs + EPSILON)
        newFeatures = np.array(rels)
        return newFeatures

    def getRankFeature(features):
        def ranks(arr):
            indices = np.argsort(arr)
            ranks = np.zeros_like(indices)
            ranks[indices] = np.arange(len(indices))[::-1]
            return ranks

        rnkSelector = [0, 1, 2, 3, 4, 5, 6, 7]
        newFeatures = np.array([ranks(features[:, rnkIdx]) for rnkIdx in rnkSelector]).T
        return newFeatures

    def getPrvFeature(features):
        newFeatures = np.concatenate([features[-1:, :], features[1:, :]], axis=0)
        return newFeatures

    def getNxtFeature(features):
        newFeatures = np.concatenate([features[1:, :], features[:1, :]], axis=0)
        return newFeatures

    features = np.array([getAudioFeature(c) for c in cliques])
    ranks = getRankFeature(features)
    rels = getRelativeFeature(features)
    prvs = getPrvFeature(features)
    nxts = getNxtFeature(features)
    features = np.concatenate([features, ranks, rels, prvs, nxts], axis=1)
    return features


class ChorusClassifier:
    def __init__(self, dataFile):
        self.dataFile = dataFile
        self.trained = False
        self.feature_names = [
            "cdur",
            "voiceRate",
            "melMedian",
            "melMin",
            "melMax",
            "head",
            "headx",
            "count",
        ]
        flen = len(self.feature_names)
        self.feature_names.extend([f"rnk_{s}" for s in self.feature_names[:flen]])
        self.feature_names.extend([f"rel_{s}" for s in self.feature_names[:flen]])
        self.feature_names.extend([f"prv_{s}" for s in self.feature_names[:flen]])
        self.feature_names.extend([f"nxt_{s}" for s in self.feature_names[:flen]])

    def train(self):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        # clf = AdaBoostClassifier(random_state=42)
        X, y = self.loadData(self.dataFile)
        clf.fit(X, y)
        self.clf = clf
        self.classes_ = clf.classes_
        self.trained = True

    def predict(self, features):
        if not self.trained:
            self.train()
        clzIdx = np.nonzero(self.classes_ == "chorus")[0][0]
        probs = self.clf.predict_proba(features)[:, clzIdx]
        if np.max(probs) < 0.5:
            # default action: select clique with maximal prob
            print(f"[WARNING]: chorus detection failed, prob:{np.max(probs)}")
            cindices = np.where(probs >= np.max(probs) - 0.05)[0]
        else:
            cindices = np.where(probs >= 0.5)[0]
        # res = self.clf.predict(features)
        # cindices = np.nonzero(res == 'chorus')[0]
        return cindices

    def loadData(self, dataFile):
        if os.path.exists(dataFile):
            with open(dataFile, "rb") as f:
                X, y = pickle.load(f)
                print(f"<{self.__class__.__name__}> load data from: {dataFile}")
                print(f'chorus/total: {sum(np.array(y)=="chorus")}/{len(y)}')
        else:
            print(f"build dataset for classifier first")
            raise FileNotFoundError(dataFile)
        return X, y


class GetAlgoData:
    def __init__(self, algorithm):
        self.algo = algorithm

    def __call__(self, dataset, idx):
        ssm_f, mels_f = getFeatures(dataset, idx)
        cliques = self.algo.getStructure(dataset, idx)
        cliques = sorted(cliques, key=lambda c: c[0])
        boundaries = np.arange(ssm_f[0].shape[0], dtype=int)
        intervals = np.array(
            [(ssm_f[0][i], ssm_f[0][i + 1]) for i in range(ssm_f[0].shape[0] - 1)]
        )
        features = getCliqueFeatures(cliques, boundaries, ssm_f[0], mels_f)
        clabels = getCliqueLabels(dataset[idx]["gt"], cliques, intervals)
        return features, clabels
