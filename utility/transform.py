import librosa
import subprocess
from third_party import msaf
import os
import pickle
import numpy as np
from mir_eval.io import load_time_series
from collections import defaultdict

from models.selfSimilarity import selfSimilarityMatrix
from models.seqRecur import cliquesFromSSM
from utility.common import extractFunctions, cliqueGroups, logSSM
from utility.dataset import Preprocess_Dataset
from configs.modelConfigs import (
    CLI_TRANSFORM_IDENTIFIER,
    MEL_SEMANTIC_LABEL_DIC,
    MEL_TRANSFORM_IDENTIFIER,
    SAMPLE_RATE,
    SSM_FEATURES,
    SSM_TRANSFORM_IDENTIFIER,
    SSM_USING_MELODY,
)
from configs.configs import logger, ALGO_BASE_DIRS


class BaseTransform:
    def __init__(self, identifier):
        self.identifier = identifier

    def preprocessor(self, wavPath, sr):
        raise NotImplementedError

    def transform(self, sample):
        raise NotImplementedError


class GenerateSSM(BaseTransform):
    def __init__(self, dataset, identifier=SSM_TRANSFORM_IDENTIFIER):
        super(GenerateSSM, self).__init__(identifier)
        tf = ExtractMel()
        self.mel_set = Preprocess_Dataset(
            tf.identifier, dataset, transform=tf.transform
        )
        self.labelSet = dataset.getLabels()
        self.labelDic = defaultdict(int, dataset.semanticLabelDic())
        assert self.labelDic["background"] == 0

    def getSSM(self, wavPath, sr):
        if SSM_USING_MELODY:
            pklPath = self.mel_set.getPklPath(-1, wavPath=wavPath)
            assert os.path.exists(
                pklPath
            ), f"{pklPath} not found, build GenerateSSM Preprocess_Dataset first"
            with open(pklPath, "rb") as f:
                mel = pickle.load(f)
            mel = (mel["times"], mel["pitches"])
            res = selfSimilarityMatrix(wavPath, mel=mel)
        else:
            res = selfSimilarityMatrix(wavPath)
        times = res["times"]
        mat = res["Ws"]
        # ['MFCCs', 'Chromas', 'Tempogram'] [Fused] ['Fused MFCC/Chroma'] ['Melody']
        ssm_features = [mat[key] for key in SSM_FEATURES]
        ssm = np.stack(ssm_features, axis=0)
        return times, ssm

    def preprocessor(self, wavPath, sr=SAMPLE_RATE):
        times, ssm = self.getSSM(wavPath, sr)
        dur = librosa.get_duration(filename=wavPath)
        assert abs(times[-1] - dur) < 0.1, f"{times[-1]} != {dur}"
        times[-1] = dur
        assert (np.diff(times, n=2) < 0.3).all(), f"{np.diff(times)[-3:]}"
        assert len(times) == ssm.shape[-1] + 1, f"{len(times)}, {ssm.shape}"
        return {"times": times, "ssm": ssm}

    def transform(self, sample):
        feature = sample["feature"]
        times, ssm = feature["times"], feature["ssm"]
        intervals, labels = sample["gt"]
        # generate target label matrix
        size = ssm.shape[-1]
        target = np.full((size, size), self.labelDic["background"])
        for label in self.labelSet:
            intvs = intervals[labels == label]
            for xbegin, xend in intvs:
                # left	a[i-1] < v <= a[i]
                # right	a[i-1] <= v < a[i]
                xlower = np.searchsorted(times, xbegin)
                xhigher = np.searchsorted(times, xend)
                for ybegin, yend in intvs:
                    ylower = np.searchsorted(times, ybegin)
                    yhigher = np.searchsorted(times, yend)
                    target[xlower:xhigher, ylower:yhigher] = self.labelDic[label]

        sample["input"] = logSSM(ssm)
        sample["target"] = target
        sample["times"] = times
        sample.pop("feature")
        return sample


class ExtractMel(BaseTransform):
    def __init__(self, identifier=MEL_TRANSFORM_IDENTIFIER):
        super(ExtractMel, self).__init__(identifier)

    def JDC(self, wavPath, output, sr=SAMPLE_RATE):
        """<Joint Detection and Classification of Singing Voice Melody Using Convolutional Recurrent Neural Networks>"""
        commands = ("python", "./melodyExtraction_JDC.py", wavPath, output)
        ret = subprocess.call(commands, cwd=ALGO_BASE_DIRS["JDC"])
        assert ret == 0, f"return value: {ret} != 0"
        times, pitches = load_time_series(output, delimiter=r"\s+|,")
        return {"times": times, "pitches": pitches}

    def SSL(self, wavPath, output, sr=SAMPLE_RATE):
        """<Semi-supervised learning using teacher-student models for vocal melody extraction>"""
        dirname = os.path.dirname(output)
        output = f"pitch_{os.path.basename(wavPath)}.txt"
        output = os.path.join(dirname, output)
        commands = ("python", "./melodyExtraction_NS.py", "-p", wavPath, "-o", dirname)
        logger.debug(f"SSL commands={commands}")
        ret = subprocess.call(commands, cwd=ALGO_BASE_DIRS["SSL"])
        assert ret == 0, f"return value: {ret} != 0"
        times, pitches = load_time_series(output, delimiter=r"\s+|,")
        return {"times": times, "pitches": pitches}

    def preprocessor(self, wavPath, sr=SAMPLE_RATE):
        wavPath = os.path.abspath(wavPath)
        title = os.path.splitext(os.path.basename(wavPath))[0]
        tmpMel = os.path.join(ALGO_BASE_DIRS["TmpDir"], f"{title}_JDC_out.csv")
        logger.debug(f"convert wav={wavPath} to mel={tmpMel}")
        return self.SSL(wavPath, tmpMel)

    def transform(self, sample):
        feature = sample["feature"]
        times, pitches = feature["times"], feature["pitches"]
        intervals, labels = sample["gt"]
        size = len(times)
        # generate target labels
        target = np.full((size,), MEL_SEMANTIC_LABEL_DIC["background"])
        for fun, idx in MEL_SEMANTIC_LABEL_DIC.items():
            ef = extractFunctions(labels, [fun])
            for intv in intervals[ef == fun]:
                lower = np.searchsorted(times, intv[0])
                higher = np.searchsorted(times, intv[1])
                target[lower:higher] = idx

        sample["input"] = pitches
        sample["times"] = times
        sample["target"] = target
        sample.pop("feature")
        return sample


class ExtractCliques(BaseTransform):
    def __init__(self, dataset, identifier=CLI_TRANSFORM_IDENTIFIER):
        super(ExtractCliques, self).__init__(identifier)
        self.tf = GenerateSSM(dataset=dataset)
        self.ssm_set = Preprocess_Dataset(
            self.tf.identifier, dataset, transform=self.tf.transform
        )

    def preprocessor(self, wavPath, sr=SAMPLE_RATE):
        pklPath = self.ssm_set.getPklPath(-1, wavPath=wavPath)
        assert os.path.exists(
            pklPath
        ), f"{pklPath} not found, build GenerateSSM Preprocess_Dataset first"
        with open(pklPath, "rb") as f:
            ssm = pickle.load(f)
        times, ssm = (ssm["times"], logSSM(ssm["ssm"])[0])
        cliques = cliquesFromSSM((times, ssm))
        return {"times": times, "cliques": cliques}

    def transform(self, sample):
        feature = sample["feature"]
        times, cliques = feature["times"], feature["cliques"]
        sample["times"] = times
        sample["cliques"] = cliques
        sample.pop("feature")
        return sample


def getFeatures(dataset, idx):
    tf = GenerateSSM(dataset=dataset)
    ssm_set = Preprocess_Dataset(tf.identifier, dataset, transform=tf.transform)
    ssmSample = ssm_set[idx]
    ssm_f = ssmSample["times"], ssmSample["input"][0]

    tf = ExtractMel()
    mel_set = Preprocess_Dataset(tf.identifier, dataset, transform=tf.transform)
    melSample = mel_set[idx]
    mels_f = melSample["times"], melSample["input"]

    return ssm_f, mels_f
