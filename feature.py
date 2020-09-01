import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from utility.dataset import *
from utility.transform import *
from utility.algorithmsWrapper import *
from models.classifier import GetAlgoData
from configs.configs import NUM_WORKERS, logger
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
        logger.info(
            f"building clique class Data for <{baseset.__class__.__name__}> @ {cpath}"
        )
        with Pool(NUM_WORKERS) as p:
            N = len(baseset)
            results = list(
                tqdm(
                    p.imap(
                        starGetCliqueClassData,
                        zip([getData] * N, [baseset] * N, range(N)),
                    ),
                    total=N,
                )
            )
        for features, clabels in results:
            X.extend([feature for feature in features])
            y.extend([clabel for clabel in clabels])
        with open(cpath, "wb") as f:
            pickle.dump((X, y), f)


def testCCDataset(method):
    logger.info(f"testCC method:{method}")
    cpath_train = CHORUS_CLASSIFIER_TRAIN_DATA_FILE[method]
    cpath_val = CHORUS_CLASSIFIER_VAL_DATA_FILE[method]
    _clf = ChorusClassifier(cpath_train)
    _clf.train()
    clf = _clf.clf
    Xt, yt = _clf.loadData(cpath_val)
    with np.printoptions(precision=3, suppress=True):
        if hasattr(clf, "feature_importances_"):
            logger.info(
                f'feature importance, {[f"{s}={x*len(_clf.feature_names):.3f}" for x, s in sorted(zip(clf.feature_importances_, _clf.feature_names))]}'
            )
        logger.info(f"test classifier on valid data, score={clf.score(Xt, yt):.3f}")


# build Preprocess Dataset for feature extraction
transforms = [
    ExtractMel(),
    GenerateSSM(),
    ExtractCliques(),
]
methods = {
    "seqRecur": GetAlgoData(AlgoSeqRecur()),
    "scluster": GetAlgoData(MsafAlgos("scluster")),
    "cnmf": GetAlgoData(MsafAlgos("cnmf")),
    "sf": GetAlgoData(MsafAlgosBdryOnly("sf")),
    "olda": GetAlgoData(MsafAlgosBdryOnly("olda")),
    "foote": GetAlgoData(MsafAlgosBdryOnly("foote")),
    "gtBoundary": GetAlgoData(GroudTruthStructure()),
}

if __name__ == "__main__":
    for tf in transforms:
        buildPreprocessDataset(USING_DATASET, tf, force=False)
    for name, getDataFun in methods.items():
        cpath_train = CHORUS_CLASSIFIER_TRAIN_DATA_FILE[name]
        cpath_val = CHORUS_CLASSIFIER_VAL_DATA_FILE[name]
        buildCCDataset(cpath_train, CLF_TRAIN_SET, getDataFun)
        buildCCDataset(cpath_val, CLF_VAL_SET, getDataFun)
        testCCDataset(name)
