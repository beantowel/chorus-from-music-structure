import glob
import os
import warnings
import numpy as np
from copy import copy

from utility.dataset import RWC_Popular_Dataset, SALAMI_Dataset

warnings.filterwarnings("ignore", ".*output shape of zoom.*")

# USING_DATASET = SALAMI_Dataset(annotation='functions')
USING_DATASET = RWC_Popular_Dataset()

# preprocess transforms
SSM_USING_MELODY = True
# <without-melody>:11 <with-melody>:12 <melody-only>:13
SSM_TRANSFORM_IDENTIFIER = 11 if not SSM_USING_MELODY else 12
MEL_TRANSFORM_IDENTIFIER = 8
BEAT_TRANSFORM_IDENTIFIER = 9
STFT_TRANSFORM_IDENTIFIER = 10
CLI_TRANSFORM_IDENTIFIER = 14

REC_SMOOTH = 9
EPSILON = 1e-9
DATASET_LABEL_DIC = USING_DATASET.getLabelDic()
SSM_SEMANTIC_LABEL_DIC = USING_DATASET.semanticLabelDic()
SSM_BACKGROUND_INDEX = len(set(SSM_SEMANTIC_LABEL_DIC.values()))
SAMPLE_RATE = 22050
SSM_TIME_STEP = 1 / SAMPLE_RATE * 512 * 10  # GraphDitty hop=512, win_fac=10
SSM_FEATURES = {11: ["Fused"], 12: ["Fused"], 13: ["Melody"],}[SSM_TRANSFORM_IDENTIFIER]

# ssm Peak algorithm
PEAK_DISTANCE_LIST = [15, 15]
PEAK_HEIGHT_LIST = [2.5, 2.0]
PEAK_PROMINENCE_LIST = [1.5, 1.5]
SSM_SEGMENT_THRESH_PERCENT = 0.9  # 0.75

# sequence recurrence algorithm
# better results with higher counts, but lower performance
CLIQUE_CANDIDATES_COUNT = 20000  # 20000
SSM_LOG_THRESH = -4.5

ADJACENT_DELTA_DISTANCE = 10
DELTA_DIS_RANGE = [5, 10, 20, 30, 40]
SMOOTH_KERNEL_SIZE = 23

MIN_STRUCTURE_COUNT = 5
MAX_CLIQUE_DURATION = 0.6

# target generation
SEMANTIC_LABEL_DIC = {
    "chorus": 1,
    "verse": 2,
}
BACKGROUND_INDEX = 0
NUM_CLASSES = max(SEMANTIC_LABEL_DIC.values()) + 1

# Chorus classifier
CLF_SPLIT_RATIO = 0.8
CLF_TRAIN_SET, CLF_VAL_SET = USING_DATASET.randomSplit(CLF_SPLIT_RATIO, seed=114514)
CHORUS_CLASSIFIER_TRAIN_DATA_FILE = {
    "seqRecur": f"data/models/{USING_DATASET.__class__.__name__}_tf{SSM_TRANSFORM_IDENTIFIER}_seqRecur_TRAIN.pkl",
    "seqRecurS": f"data/models/{USING_DATASET.__class__.__name__}_tf{SSM_TRANSFORM_IDENTIFIER}_seqRecurS_TRAIN.pkl",
    # 'vmo': f'data/models/{USING_DATASET.__class__.__name__}_vmo_TRAIN.pkl',
    "scluster": f"data/models/{USING_DATASET.__class__.__name__}_scluster_TRAIN.pkl",
    # 'fmc2d': f'data/models/{USING_DATASET.__class__.__name__}_fmc2d_TRAIN.pkl',
    "cnmf": f"data/models/{USING_DATASET.__class__.__name__}_cnmf_TRAIN.pkl",
    "sf": f"data/models/{USING_DATASET.__class__.__name__}_sf_TRAIN.pkl",
    "olda": f"data/models/{USING_DATASET.__class__.__name__}_olda_TRAIN.pkl",
    "foote": f"data/models/{USING_DATASET.__class__.__name__}_foote_TRAIN.pkl",
    "gt": f"data/models/{USING_DATASET.__class__.__name__}_gt_TRAIN.pkl",
}
CHORUS_CLASSIFIER_VAL_DATA_FILE = {
    key: val.replace("TRAIN.pkl", "VAL.pkl")
    for key, val in CHORUS_CLASSIFIER_TRAIN_DATA_FILE.items()
}
# clique class target generation
CC_PRECISION = 0.5
CC_RECALL = 0.3
MINIMUM_CHORUS_DUR = 10
