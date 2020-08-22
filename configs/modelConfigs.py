import os
import numpy as np
from copy import copy

from utility.dataset import RWC_Popular_Dataset, SALAMI_Dataset

# USING_DATASET = SALAMI_Dataset(annotation='functions')
USING_DATASET = RWC_Popular_Dataset()

# preprocess transforms
SSM_USING_MELODY = True
# <without-melody>:11 <with-melody>:12 <melody-only>:13
SSM_TRANSFORM_IDENTIFIER = 11 if not SSM_USING_MELODY else 12
MEL_TRANSFORM_IDENTIFIER = 8
# BEAT_TRANSFORM_IDENTIFIER = 9
# STFT_TRANSFORM_IDENTIFIER = 10
CLI_TRANSFORM_IDENTIFIER = 14

# similarity fusion settings
REC_SMOOTH = 9
EPSILON = 1e-9
DATASET_LABEL_DIC = USING_DATASET.getLabelDic()
SSM_SEMANTIC_LABEL_DIC = USING_DATASET.semanticLabelDic()
SSM_BACKGROUND_INDEX = len(set(SSM_SEMANTIC_LABEL_DIC.values()))
SAMPLE_RATE = 22050
SSM_TIME_STEP = 1 / SAMPLE_RATE * 512 * 10  # GraphDitty hop=512, win_fac=10
SSM_FEATURES = {11: ["Fused"], 12: ["Fused"], 13: ["Melody"],}[SSM_TRANSFORM_IDENTIFIER]

# sequence recurrence algorithm
SSM_LOG_THRESH = -4.5

ADJACENT_DELTA_DISTANCE = 10
DELTA_DIS_RANGE = [5, 10, 20]
DELTA_BLOCK_RANGE = [0]
SMOOTH_KERNEL_SIZE = 23
FALSE_POSITIVE_ERROR = 0.1

MIN_STRUCTURE_COUNT = 5
# MAX_CLIQUE_DURATION = 0.6

# target generation
SEMANTIC_LABEL_DIC = {
    "chorus": 1,
    "verse": 2,
}
BACKGROUND_INDEX = 0

# Chorus classifier
CLF_SPLIT_RATIO = 0.8
CLF_TRAIN_SET, CLF_VAL_SET = USING_DATASET.randomSplit(CLF_SPLIT_RATIO, seed=114514)
CHORUS_CLASSIFIER_TRAIN_DATA_FILE = {
    "seqRecur": f"data/models/{USING_DATASET.__class__.__name__}_tf{SSM_TRANSFORM_IDENTIFIER}_seqRecur_TRAIN.pkl",
    "scluster": f"data/models/{USING_DATASET.__class__.__name__}_scluster_TRAIN.pkl",
    "cnmf": f"data/models/{USING_DATASET.__class__.__name__}_cnmf_TRAIN.pkl",
    "sf": f"data/models/{USING_DATASET.__class__.__name__}_sf_TRAIN.pkl",
    "olda": f"data/models/{USING_DATASET.__class__.__name__}_olda_TRAIN.pkl",
    "foote": f"data/models/{USING_DATASET.__class__.__name__}_foote_TRAIN.pkl",
    "gtBoundary": f"data/models/{USING_DATASET.__class__.__name__}_gtBoundary_TRAIN.pkl",
}
CHORUS_CLASSIFIER_VAL_DATA_FILE = {
    key: val.replace("TRAIN.pkl", "VAL.pkl")
    for key, val in CHORUS_CLASSIFIER_TRAIN_DATA_FILE.items()
}
# clique class target generation
CC_PRECISION = 0.5
CC_RECALL = 0.1
MINIMUM_CHORUS_DUR = 10
