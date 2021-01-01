import os
import time
import pickle
import librosa
import time
import unicodedata
import numpy as np
from copy import deepcopy, copy
from itertools import chain
from collections import namedtuple
from multiprocessing import Pool
from tqdm import tqdm
from mir_eval.io import load_labeled_events, load_labeled_intervals

from configs.configs import DATASET_BASE_DIRS, NUM_WORKERS, logger


StructDataPathPair = namedtuple("StructDataPathPair", "title wav GT")


def convertFileName(basedir, norm="NFD"):
    files = os.listdir(basedir)
    for fileName in files:
        normalName = unicodedata.normalize(norm, fileName)
        src = os.path.join(basedir, fileName)
        dst = os.path.join(basedir, normalName)
        if src != dst:
            os.rename(src, dst)
            logger.info(f"rename, src='{src}' dst='{dst}'")


class BaseStructDataset:
    def __init__(self, baseDir, transform):
        self.baseDir = baseDir
        self.transform = transform
        self.pathPairs = []  # <StructDataPathPair> List
        self.labelSet = None

    def __len__(self):
        return len(self.pathPairs)

    def __getitem__(self, idx):
        return self.getSample(self.pathPairs[idx])

    def getSample(self, pathPair):
        wavPath = pathPair.wav
        GTPath = pathPair.GT
        title = pathPair.title

        intervals, labels = self.loadGT(GTPath)
        labels = np.array(labels)
        # force gt intervals in range (0, AudioDuration)
        # assert intervals[0][0] == 0, f'{GTPath}'
        intervals[0][0] = 0
        dur = librosa.get_duration(filename=wavPath)
        while intervals[-1][0] >= dur:
            intervals = intervals[:-1]
            labels = labels[:-1]
        intervals[-1][1] = dur
        # mirex format: labeled intervals
        gt = (intervals, labels)

        sample = {"wavPath": wavPath, "gt": gt, "title": title}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def getLabels(self, refresh=False):
        # all distinct labels in the dataset
        if refresh or self.labelSet is None:
            labelSet = set()
            for pair in self.pathPairs:
                _, labels = self.loadGT(pair.GT)
                labelSet = labelSet.union(labels)
            labelSet = sorted(labelSet)
        return labelSet

    def loadGT(self, GTPath):
        raise NotImplementedError

    def semanticLabelDic(self):
        # custom semantic index for labels in the dataset, 0 IS RESERVED, DO NOT USE!!
        raise NotImplementedError

    def randomSplit(self, splitRatio, seed):
        np.random.seed(seed)
        indices = np.random.permutation(range(len(self)))
        newLen = int(len(self) * splitRatio)
        a_pathPairs = [self.pathPairs[i] for i in indices[:newLen]]
        b_pathPairs = [self.pathPairs[i] for i in indices[newLen:]]

        a, b = deepcopy(self), deepcopy(self)
        a.pathPairs = a_pathPairs
        b.pathPairs = b_pathPairs
        return a, b


class SALAMI_Dataset(BaseStructDataset):
    """<Design and creation of a large-scale database of structural annotations>"""

    def __init__(
        self,
        baseDir=DATASET_BASE_DIRS["SALAMI"],
        annotation="functions",
        singleAnnt=True,
        transform=None,
    ):
        super(SALAMI_Dataset, self).__init__(baseDir, transform)
        # collections/<title>.mp3
        # annotations/<title>/parsed/textfile<annotator_number>_<annotation>.txt
        self.annotation = annotation
        validAnnotations = ["functions", "lowercase", "uppercase"]
        if annotation not in validAnnotations:
            raise ValueError(f"annotation '{annotation}' not in {validAnnotations}")

        for filename in sorted(os.listdir(os.path.join(baseDir, "collections"))):
            title = os.path.splitext(filename)[0]
            assert (
                os.path.splitext(filename)[1] == ".mp3"
            ), f"{filename}: wrong extension"
            wavPath = os.path.join(baseDir, "collections", filename)
            GTPath = os.path.join(
                baseDir, "annotations", title, "parsed", f"textfile1_{annotation}.txt"
            )
            GTPath2 = os.path.join(
                baseDir, "annotations", title, "parsed", f"textfile2_{annotation}.txt"
            )
            self._addPath(GTPath, title, wavPath)
            # consider both annotation as ground truth, even they conflict with each other sometimes (around 50%!)
            if not singleAnnt:
                self._addPath(GTPath2, title, wavPath)

    def loadGT(self, GTPath):
        """load melody ground truth"""
        e_times, labels = load_labeled_events(GTPath, delimiter="\t")
        intervals = np.array([e_times[:-1], e_times[1:]]).T
        # iganore variations like <A, A'>
        labels = [lab.strip("'") for lab in labels[:-1]]
        # merge <silence> and <Silence>
        labels = ["Silence" if lab == "silence" else lab for lab in labels]
        # remove negative duration intervals
        positiveIntv = np.array([intv[1] - intv[0] > 0 for intv in intervals])
        intervals = intervals[positiveIntv]
        labels = np.array(labels)[positiveIntv]
        return intervals, labels

    def _addPath(self, GTPath, title, wavPath):
        if os.path.exists(GTPath):
            _, labels = self.loadGT(GTPath)
            if "Chorus" in labels:
                self.pathPairs.append(StructDataPathPair(title, wavPath, GTPath))

    def semanticLabelDic(self):
        if self.annotation == "functions":
            dic = {
                "&pause": 33,
                "Bridge": 1,
                "Chorus": 2,
                "Coda": 3,
                "End": 4,
                "Fade-out": 5,
                "Head": 6,
                "Instrumental": 7,
                "Interlude": 8,
                "Intro": 9,
                "Main_Theme": 10,
                "Outro": 11,
                "Pre-Chorus": 12,
                "Pre-Verse": 13,
                "Silence": 14,
                "Solo": 15,
                "Theme": 16,
                "Transition": 17,
                "Verse": 18,
                "applause": 19,
                "banjo": 20,
                "break": 21,
                "build": 22,
                "crowd_sounds": 23,
                "no_function": 24,
                "post-chorus": 25,
                "post-verse": 26,
                "spoken_voice": 27,
                "stage_sounds": 28,
                "stage_speaking": 29,
                "tag": 30,
                "variation": 31,
                "voice": 32,
            }
        elif self.annotation == "uppercase":
            dic = {
                "A": 2,
                "B": 2,
                "C": 2,
                "D": 2,
                "E": 2,
                "F": 2,
                "G": 2,
                "H": 2,
                "I": 2,
                "J": 2,
                "K": 2,
                "L": 2,
                "M": 2,
                "N": 2,
                "O": 2,
                "Q": 2,
                "R": 2,
                "S": 2,
                "Silence": 1,
                "T": 2,
                "U": 2,
                "Y": 2,
                "Z": 2,
            }
        elif self.annotation == "lowercase":
            dic = {
                "Silence": 1,
                "a": 2,
                "b": 2,
                "c": 2,
                "d": 2,
                "e": 2,
                "f": 2,
                "g": 2,
                "h": 2,
                "i": 2,
                "j": 2,
                "k": 2,
                "l": 2,
                "m": 2,
                "n": 2,
                "o": 2,
                "p": 2,
                "q": 2,
                "r": 2,
                "s": 2,
                "t": 2,
                "u": 2,
                "v": 2,
                "w": 2,
                "x": 2,
                "y": 2,
                "z": 2,
            }
        return dic


class RWC_Popular_Dataset(BaseStructDataset):
    """AIST Annotation for RWC Music Database"""

    def __init__(self, baseDir=DATASET_BASE_DIRS["RWC"], transform=None):
        super(RWC_Popular_Dataset, self).__init__(baseDir, transform)
        # RWC-MDB-P-2001/AIST.RWC-MDB-P-2001.CHORUS/RM-P<Num:03d>.CHORUS.TXT
        # RWC-MDB-P-2001/RWC研究用音楽データベース[| Disc <Id>]/<Num':02d> <title>.wav
        discPaths = [os.path.join(baseDir, "RWC-MDB-P-2001", "RWC研究用音楽データベース")]
        for Id in range(2, 8):
            dp = os.path.join(baseDir, "RWC-MDB-P-2001", f"RWC研究用音楽データベース Disc {Id}")
            discPaths.append(dp)
        self._addPairFromPaths(discPaths, "P")

    def _addPairFromPaths(self, discPaths, X):
        def listDisc(discPath):
            # ensure the wav files are well-ordered, or GTfile will mismatch
            names = sorted(os.listdir(discPath))
            return [os.path.join(discPath, n) for n in names]

        for num, wavPath in enumerate(chain(*map(listDisc, discPaths))):
            _, tail = os.path.split(wavPath)
            assert tail[-4:] == ".wav", "has non-wave file in the RWC disc folder"
            title = tail[:-4]
            GTPath = os.path.join(
                self.baseDir,
                f"RWC-MDB-{X}-2001",
                f"AIST.RWC-MDB-{X}-2001.CHORUS",
                f"RM-{X}{num+1:03d}.CHORUS.TXT",
            )
            if unicodedata.normalize("NFD", title) != title:
                logger.warn(f"file={wavPath} titile={title} is not in unicode NFD form")
            self.pathPairs.append(StructDataPathPair(title, wavPath, GTPath))

    def loadGT(self, GTPath):
        intervals, labels = load_labeled_intervals(GTPath, delimiter="\t")
        intervals = intervals / 100.0
        # ignore pitch shift like: <"chorus A"   (-10)>
        labels = [label.split("\t")[0].strip('"') for label in labels]
        return intervals, labels

    def semanticLabelDic(self):
        dic = {
            "bridge A": 3,
            "bridge B": 3,
            "bridge C": 3,
            "bridge D": 3,
            "chorus A": 1,
            "chorus B": 1,
            "chorus C": 1,
            "chorus D": 1,
            "ending": 3,
            "intro": 3,
            "nothing": 3,
            "post-chorus": 3,
            "pre-chorus": 3,
            "verse A": 2,
            "verse B": 2,
            "verse C": 2,
        }
        return dic


class RWC_Popular_Dataset_accomp(RWC_Popular_Dataset):
    def __init__(self, accompDir=DATASET_BASE_DIRS["RWC_accomp"], transform=None):
        super(RWC_Popular_Dataset_accomp, self).__init__(transform=transform)
        convertFileName(accompDir)
        self.accompDir = accompDir
        newPathPairs = []
        for pair in self.pathPairs:
            title = pair.title
            gt = pair.GT
            accomp = os.path.join(accompDir, title, "accompaniment.wav")
            wav = os.path.join(accompDir, title, f"{title}.wav")
            if os.path.exists(accomp):
                os.rename(accomp, wav)
                logger.warn(f"rename {accomp} -> {wav}")
            newPathPairs.append(StructDataPathPair(title, wav, gt))
        self.pathPairs = newPathPairs


class CCM_Dataset(BaseStructDataset):
    """Dataset built by China Conservatory of Music"""

    def __init__(self, baseDir=DATASET_BASE_DIRS["CCM"], transform=None):
        super(CCM_Dataset, self).__init__(baseDir, transform)
        # chorus/<title>.txt
        # audio/<title>.mp3
        for filename in sorted(os.listdir(os.path.join(baseDir, "audio"))):
            title = os.path.splitext(filename)[0]
            assert os.path.splitext(filename)[1] in [
                ".mp3",
                ".flac",
            ], f"{filename}: wrong extension"
            wavPath = os.path.join(baseDir, "audio", filename)
            GTPath = os.path.join(baseDir, "chorus", title + ".txt")
            NFDTitle = unicodedata.normalize("NFD", title)
            if NFDTitle != title:
                logger.warn(
                    f"filename={wavPath} is not in unicode NFD form, expected={NFDTitle}"
                )
            self.pathPairs.append(StructDataPathPair(title, wavPath, GTPath))

    def loadGT(self, GTPath):
        intervals, labels = load_labeled_intervals(GTPath, delimiter="\t")
        labels = np.array([label.strip('"') for label in labels])
        intervals = intervals / 100.0
        return intervals, labels

    def semanticLabelDic(self):
        dic = {
            "Bridge": 3,
            "Bridge 1": 3,
            "Bridge 2": 3,
            "Bridge A": 3,
            "Chorus": 1,
            "Chorus A": 1,
            "Chorus B": 1,
            "Chorus C": 1,
            "Chorus D": 1,
            "Chorus E": 1,
            "Chorus F": 1,
            "Chorus G": 1,
            "Chorus H": 1,
            "Chorus I": 1,
            "Ending": 3,
            "Interlude": 3,
            "Interlude A": 3,
            "Interlude B": 3,
            "Intro": 3,
            "Ore-chorus B": 3,
            "Post-chorus A": 3,
            "Post-chorus B": 3,
            "Pre-chorus A": 3,
            "Pre-chorus B": 3,
            "Pre-chorus C": 3,
            "Re-intro": 3,
            "Re-intro A": 3,
            "Re-intro B": 3,
            "Re-intro C": 3,
            "Reintro": 3,
            "Verse": 2,
            "Verse A": 2,
            "Verse B": 2,
            "Verse C": 2,
            "Verse D": 2,
            "Verse D(Ending)": 2,
            "Verse E": 2,
            "Verse F": 2,
            "bridge": 3,
            "interlude B": 3,
        }
        return dic


class Huawei_Dataset(BaseStructDataset):
    def __init__(self, baseDir=DATASET_BASE_DIRS["Huawei"], transform=None):
        super(Huawei_Dataset, self).__init__(baseDir, transform)
        # struct/<title>.txt
        # audio/<title>.mp3
        for filename in sorted(os.listdir(os.path.join(baseDir, "audio"))):
            title = os.path.splitext(filename)[0]
            assert (
                os.path.splitext(filename)[1] == ".mp3"
            ), f"{filename}: wrong extension"
            wavPath = os.path.join(baseDir, "audio", filename)
            GTPath = os.path.join(baseDir, "struct", title + ".txt")
            NFDTitle = unicodedata.normalize("NFD", title)
            if NFDTitle != title:
                logger.warn(
                    f"filename={wavPath} is not in unicode NFD form, expected={NFDTitle}"
                )
            _, labels = self.loadGT(GTPath)
            if "chorus" in labels:
                self.pathPairs.append(StructDataPathPair(title, wavPath, GTPath))
            else:
                logger.warn(f"no chorus section, file={GTPath}")

    def loadGT(self, GTPath):
        intervals, labels = load_labeled_intervals(GTPath, delimiter="\t")
        return intervals, labels

    def semanticLabelDic(self):
        dic = {
            "chorus": 1,
            "verse": 2,
            "others": 3,
        }
        return dic


class Preprocess_Dataset:
    def __init__(
        self,
        tid,
        dataset,
        baseDir=DATASET_BASE_DIRS["LocalTemporary_Dataset"],
        transform=None,
    ):
        self.baseDir = baseDir
        self.tid = tid  # unique transform id to distinguish from same filename
        self.dataset = dataset
        self.ddir = os.path.join(baseDir, dataset.__class__.__name__)
        self.transform = transform
        assert isinstance(
            dataset, BaseStructDataset
        ), f"{type(dataset)} is not {type(BaseStructDataset)}"
        if not os.path.exists(self.ddir):
            os.mkdir(self.ddir)

    def build(self, preprocessor, force=False, num_workers=NUM_WORKERS):
        logger.info(
            f"building <{self.__class__.__name__}> from <{self.dataset.__class__.__name__}> with transform identifier=<{self.tid}>"
        )
        self.preprocessor = preprocessor
        self.force_build = force
        with Pool(num_workers) as p:
            N = len(self.dataset)
            _ = list(tqdm(p.imap(self.storeFeature, range(N)), total=N))

    def storeFeature(self, i):
        # <ddir>/<orig_name>-<id>.pkl
        pklPath = self.getPklPath(i)
        if (not os.path.exists(pklPath)) or self.force_build:
            feature = self.preprocessor(self.dataset.pathPairs[i].wav)
            with open(pklPath, "wb") as f:
                pickle.dump(feature, f, pickle.HIGHEST_PROTOCOL)

    def loadFeature(self, i):
        # <ddir>/<orig_name>-<id>.pkl
        pklPath = self.getPklPath(i)
        try:
            with open(pklPath, "rb") as f:
                feature = pickle.load(f)
            return feature
        except FileNotFoundError as e:
            logger.error(f'file "{pklPath}" not found, build the dataset first.')
            raise e

    def getPklPath(self, idx, wavPath=None):
        if wavPath is None:
            wavPath = self.dataset.pathPairs[idx].wav
        _, filename = os.path.split(wavPath)
        orig_name, _ = os.path.splitext(filename)
        new_path = os.path.join(self.ddir, f"{orig_name}-{self.tid}.pkl")
        return new_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample["feature"] = self.loadFeature(idx)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def buildPreprocessDataset(dataset, tf, force=False):
    preDataset = Preprocess_Dataset(tf.identifier, dataset)
    preDataset.build(tf.preprocessor, force=force)
    return preDataset


class DummyDataset(BaseStructDataset):
    def __init__(self, audioList):
        super(DummyDataset, self).__init__(baseDir=None, transform=None)
        for wavPath in audioList:
            title = os.path.splitext(os.path.split(wavPath)[-1])[0]
            self.pathPairs.append(StructDataPathPair(title, wavPath, None))

    def getSample(self, pathPair):
        wavPath = pathPair.wav
        title = pathPair.title

        gt = (np.array([], dtype="U16"), np.array([], dtype="U16"))

        sample = {"wavPath": wavPath, "gt": gt, "title": title}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def loadGT(self, GTPath):
        intervals = np.array(
            [
                (0, 0),
            ]
        )
        labels = ["unknown"]
        return intervals, labels

    def semanticLabelDic(self):
        dic = {"unkonwn": 0}
        return dic
