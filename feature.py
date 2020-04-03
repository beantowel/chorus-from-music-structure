import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from utility.dataset import *
from utility.transform import *
from utility.algorithmsWrapper import *
from models.classifier import GetAlgoData
from configs.configs import NUM_WORKERS
from configs.modelConfigs import *
from models.classifier import ChorusClassifier


def buildPreprocessDataset(dataset, tf, force=False):
    preDataset = Preprocess_Dataset(tf.identifier, dataset)
    preDataset.build(tf.preprocessor, force=force)
    return preDataset


def starGetCliqueClassData(t):
    getData, baseset, idx = t
    res = getData(baseset, idx)
    return res


def buildCCDataset(cpath, baseset, getData, force=True):
    if not os.path.exists(cpath) or force:
        X = []
        y = []
        print(
            f'building clique class Data for <{baseset.__class__.__name__}> @ {cpath}')
        with Pool(NUM_WORKERS) as p:
            N = len(baseset)
            results = list(
                tqdm(
                    p.imap(
                        starGetCliqueClassData,
                        zip(
                            [getData]*N,
                            [baseset]*N,
                            range(N))),
                    total=N))
        for features, clabels in results:
            X.extend([feature for feature in features])
            y.extend([clabel for clabel in clabels])
        with open(cpath, 'wb') as f:
            pickle.dump((X, y), f)


def testCCDataset(method):
    print(f'testCC method:{method}')
    cpath_train = CHORUS_CLASSIFIER_TRAIN_DATA_FILE[method]
    cpath_val = CHORUS_CLASSIFIER_VAL_DATA_FILE[method]
    _clf = ChorusClassifier(cpath_train)
    _clf.train()
    clf = _clf.clf
    Xt, yt = _clf.loadData(cpath_val)
    with np.printoptions(precision=3, suppress=True):
        if hasattr(clf, 'feature_importances_'):
            print(
                f'importance: {[f"{s}:{x:.3f}" for x, s in zip(clf.feature_importances_, _clf.feature_names)]}')
        print(f'score:{clf.score(Xt, yt):.3f}')


# build Preprocess Dataset for feature extraction
transforms = [ExtractMel(), GenerateSSM(), ExtractCliques()]
methods = {
    # 'seqRecur': GetAlgoData(AlgoSeqRecur()),
    'seqRecurS': GetAlgoData(AlgoSeqRecurSingle()),
    # 'scluster': GetAlgoData(MsafAlgos('scluster')),
    # 'cnmf': GetAlgoData(MsafAlgos('cnmf')),
    # 'sf': GetAlgoData(MsafAlgosBdryOnly('sf')),
    # 'olda': GetAlgoData(MsafAlgosBdryOnly('olda')),
    # 'foote': GetAlgoData(MsafAlgosBdryOnly('foote')),
    # 'gt': GetAlgoData(GroudTruthStructure()),
    # 'vmo': GetAlgoData(MsafAlgos('vmo')),
}

if __name__ == '__main__':
    for tf in transforms:
        buildPreprocessDataset(USING_DATASET, tf, force=False)
    for name, fun in methods.items():
        cpath_train = CHORUS_CLASSIFIER_TRAIN_DATA_FILE[name]
        cpath_val = CHORUS_CLASSIFIER_VAL_DATA_FILE[name]
        getData = fun
        buildCCDataset(cpath_train, CLF_TRAIN_SET, getData)
        buildCCDataset(cpath_val, CLF_VAL_SET, getData)
        testCCDataset(name)
