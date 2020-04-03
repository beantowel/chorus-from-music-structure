import os
import time
import pickle
import librosa
import time
import numpy as np
from queue import deque
from copy import deepcopy, copy
from itertools import chain
from collections import namedtuple, deque
from multiprocessing import Pool
from tqdm import tqdm
from mir_eval.io import load_labeled_events, load_labeled_intervals

from configs.configs import DATASET_BASE_DIRS, NUM_WORKERS


StructDataPathPair = namedtuple('StructDataPathPair', 'title wav GT')


class BaseStructDataset():

    def __init__(self, baseDir, transform):
        self.baseDir = baseDir
        self.transform = transform
        self.pathPairs = []  # <StructDataPathPair> List
        self.labelDic = None

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

        sample = {'wavPath': wavPath, 'gt': gt, 'title': title}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def getLabelDic(self, refresh=False):
        if refresh or self.labelDic is None:
            labelSet = set()
            for pair in self.pathPairs:
                _, labels = self.loadGT(pair.GT)
                labelSet = labelSet.union(labels)
            labelSet = sorted(labelSet)
            self.labelDic = {label: i for i,
                             label in enumerate(labelSet)}
        return self.labelDic

    def loadGT(self, GTPath):
        raise NotImplementedError

    def semanticLabelDic(self):
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
    '''<Design and creation of a large-scale database of structural annotations>'''

    def __init__(self, baseDir=DATASET_BASE_DIRS['SALAMI'], annotation='functions', singleAnnt=True, transform=None):
        super(SALAMI_Dataset, self).__init__(baseDir, transform)
        # collections/<title>.mp3
        # annotations/<title>/parsed/textfile<annotator_number>_<annotation>.txt
        self.annotation = annotation
        validAnnotations = ['functions', 'lowercase', 'uppercase']
        if annotation not in validAnnotations:
            raise ValueError(
                f"annotation '{annotation}' not in {validAnnotations}")

        for filename in sorted(os.listdir(os.path.join(baseDir, 'collections'))):
            title = os.path.splitext(filename)[0]
            wavPath = os.path.join(baseDir, 'collections', title+'.mp3')
            GTPath = os.path.join(baseDir, 'annotations',
                                  title, 'parsed', f'textfile1_{annotation}.txt')
            GTPath2 = os.path.join(baseDir, 'annotations',
                                   title, 'parsed', f'textfile2_{annotation}.txt')
            self.addPath(GTPath, title, wavPath)
            # consider both annotation as ground truth, even they conflict with each other sometimes (around 50%!)
            if not singleAnnt:
                self.addPath(GTPath2, title, wavPath)

    def loadGT(self, GTPath):
        '''load melody ground truth'''
        e_times, labels = load_labeled_events(GTPath, delimiter='\t')
        intervals = np.array([e_times[:-1], e_times[1:]]).T
        # iganore variations like <A, A'>
        labels = [lab.strip("'") for lab in labels[:-1]]
        # merge <silence> and <Silence>
        labels = ['Silence' if lab == 'silence' else lab for lab in labels]
        # remove negative duration intervals
        posDuration = np.array([intv[1] - intv[0] > 0 for intv in intervals])
        intervals = intervals[posDuration]
        labels = np.array(labels)[posDuration]
        return intervals, labels

    def addPath(self, GTPath, title, wavPath):
        if os.path.exists(GTPath):
            intervals, labels = self.loadGT(GTPath)
            if 'Chorus' in labels:
                self.pathPairs.append(
                    StructDataPathPair(title, wavPath, GTPath))

    def semanticLabelDic(self):
        if self.annotation == 'functions':
            dic = {'&pause': 0, 'Bridge': 1, 'Chorus': 2, 'Coda': 3, 'End': 4, 'Fade-out': 5, 'Head': 6, 'Instrumental': 7, 'Interlude': 8, 'Intro': 9, 'Main_Theme': 10, 'Outro': 11, 'Pre-Chorus': 12, 'Pre-Verse': 13, 'Silence': 14, 'Solo': 15, 'Theme': 16, 'Transition': 17,
                   'Verse': 18, 'applause': 19, 'banjo': 20, 'break': 21, 'build': 22, 'crowd_sounds': 23, 'no_function': 24, 'post-chorus': 25, 'post-verse': 26, 'spoken_voice': 27, 'stage_sounds': 28, 'stage_speaking': 29, 'tag': 30, 'variation': 31, 'voice': 32}
        elif self.annotation == 'uppercase':
            dic = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0,
                   'M': 0, 'N': 0, 'O': 0, 'Q': 0, 'R': 0, 'S': 0, 'Silence': 1, 'T': 0, 'U': 0, 'Y': 0, 'Z': 0}
        elif self.annotation == 'lowercase':
            dic = {'Silence': 1, 'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0, 'g': 0, 'h': 0, 'i': 0, 'j': 0, 'k': 0, 'l': 0,
                   'm': 0, 'n': 0, 'o': 0, 'p': 0, 'q': 0, 'r': 0, 's': 0, 't': 0, 'u': 0, 'v': 0, 'w': 0, 'x': 0, 'y': 0, 'z': 0}
        return dic


class RWC_Popular_Dataset(BaseStructDataset):
    '''AIST Annotation for RWC Music Database'''

    def __init__(self, baseDir=DATASET_BASE_DIRS['RWC'], transform=None):
        super(RWC_Popular_Dataset, self).__init__(baseDir, transform)
        # RWC-MDB-P-2001/AIST.RWC-MDB-P-2001.MELODY/RM-P<Num:03d>.MELODY.TXT
        # RWC-MDB-P-2001/RWC研究用音楽データベース[| Disc <Id>]/<Num':02d> <title>.wav
        discPaths = [os.path.join(
            baseDir, 'RWC-MDB-P-2001', 'RWC研究用音楽データベース')]
        for Id in range(2, 8):
            dp = os.path.join(baseDir, 'RWC-MDB-P-2001',
                              f'RWC研究用音楽データベース Disc {Id}')
            discPaths.append(dp)
        self.addPairFromPaths(discPaths, 'P')

    def addPairFromPaths(self, discPaths, X):
        def listDisc(discPath):
            # ensure the wav files are well-ordered, or GTfile will mismatch
            names = sorted(os.listdir(discPath))
            return [os.path.join(discPath, n) for n in names]

        for num, wavPath in enumerate(chain(*map(listDisc, discPaths))):
            _, tail = os.path.split(wavPath)
            assert tail[-4:] == '.wav', 'has non-wave file in the RWC disc folder'
            title = tail[:-4]
            GTPath = os.path.join(
                self.baseDir, f'RWC-MDB-{X}-2001', f'AIST.RWC-MDB-{X}-2001.CHORUS', f'RM-{X}{num+1:03d}.CHORUS.TXT')
            self.pathPairs.append(StructDataPathPair(title, wavPath, GTPath))

    def loadGT(self, GTPath):
        intervals, labels = load_labeled_intervals(GTPath, delimiter='\t')
        intervals = intervals / 100.
        # ignore pitch shift like: <"chorus A"   (-10)>
        labels = [label.split('\t')[0].strip('"') for label in labels]
        return intervals, labels

    def semanticLabelDic(self):
        dic = {'bridge A': 0, 'bridge B': 0, 'bridge C': 0, 'bridge D': 0, 'chorus A': 1, 'chorus B': 1, 'chorus C': 1, 'chorus D': 1,
               'ending': 0, 'intro': 0, 'nothing': 0, 'post-chorus': 0, 'pre-chorus': 0, 'verse A': 2, 'verse B': 2, 'verse C': 2}
        return dic


class Preprocess_Dataset():

    def __init__(self, tid, dataset, baseDir=DATASET_BASE_DIRS['LocalTemporary_Dataset'], transform=None):
        self.baseDir = baseDir
        self.tid = tid  # unique transform id to distinguish from same filename
        self.dataset = dataset
        self.ddir = os.path.join(baseDir, dataset.__class__.__name__)
        self.transform = transform
        assert isinstance(
            dataset, BaseStructDataset), f'{type(dataset)} is not {type(BaseStructDataset)}'
        if not os.path.exists(self.ddir):
            os.mkdir(self.ddir)

    def build(self, preprocessor, force=False, num_workers=NUM_WORKERS):
        print(
            f'building <{self.__class__.__name__}> from <{self.dataset.__class__.__name__}> with transform identifier=<{self.tid}>')
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
            with open(pklPath, 'wb') as f:
                pickle.dump(feature, f, pickle.HIGHEST_PROTOCOL)

    def loadFeature(self, i):
        # <ddir>/<orig_name>-<id>.pkl
        pklPath = self.getPklPath(i)
        try:
            with open(pklPath, 'rb') as f:
                feature = pickle.load(f)
            return feature
        except FileNotFoundError as e:
            print(f'"{pklPath}" not found, build the dataset first.')
            raise e

    def getPklPath(self, idx, wavPath=None):
        if wavPath is None:
            wavPath = self.dataset.pathPairs[idx].wav
        _, filename = os.path.split(wavPath)
        orig_name, _ = os.path.splitext(filename)
        new_path = os.path.join(self.ddir, f'{orig_name}-{self.tid}.pkl')
        return new_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample['feature'] = self.loadFeature(idx)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class DummyDataset(BaseStructDataset):

    def __init__(self, audioList):
        super(DummyDataset, self).__init__(baseDir=None, transform=None)
        for wavPath in audioList:
            title = os.path.splitext(os.path.split(wavPath)[-1])[0]
            self.pathPairs.append(
                StructDataPathPair(title, wavPath, None))

    def getSample(self, pathPair):
        wavPath = pathPair.wav
        title = pathPair.title

        gt = (np.array([]), np.array([]))

        sample = {'wavPath': wavPath, 'gt': gt, 'title': title}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
