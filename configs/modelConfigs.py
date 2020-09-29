import os
import numpy as np
from copy import copy
from collections import defaultdict

from utility.dataset import RWC_Popular_Dataset, SALAMI_Dataset, CCM_Dataset

# USING_DATASET = SALAMI_Dataset(annotation='functions')
# USING_DATASET = RWC_Popular_Dataset()
USING_DATASET = CCM_Dataset()


# preprocess transforms
SSM_USING_MELODY = True
MEL_TRANSFORM_IDENTIFIER = 8
SSM_TRANSFORM_IDENTIFIER = (
    11 if not SSM_USING_MELODY else 12
)  # <without-melody>:11 <with-melody>:12 <melody-only>:13
CLI_TRANSFORM_IDENTIFIER = 15 if not SSM_USING_MELODY else 14

# similarity fusion settings
REC_SMOOTH = 9
EPSILON = 1e-9
SAMPLE_RATE = 22050
SSM_TIME_STEP = 1 / SAMPLE_RATE * 512 * 10  # GraphDitty hop=512, win_fac=10
SSM_FEATURES = {
    11: ["Fused"],
    12: ["Fused"],
    13: ["Melody"],
}[SSM_TRANSFORM_IDENTIFIER]

# pitch chroma feature
PITCH_CHROMA_CLASS = 24
PITCH_CHROMA_STEP = 10
PITCH_CHROMA_SEQ = 5
# PITCH_CHROMA_CLASS = int(os.getenv("PC_CLASS"))
# PITCH_CHROMA_STEP = int(os.getenv("PC_STEP"))
# PITCH_CHROMA_SEQ = int(os.getenv("PC_SEQ"))

# sequence recurrence algorithm
SSM_LOG_THRESH = -4.5
ADJACENT_DELTA_DISTANCE = 10
DELTA_DIS_RANGE = [5, 10, 20]
SMOOTH_KERNEL_SIZE = 23
SMOOTH_KERNEL_SIZE_RANGE = [23, 31, 47]
FALSE_POSITIVE_ERROR = 0.15
MIN_STRUCTURE_COUNT = 5
# MAX_CLIQUE_DURATION = 0.6

# boundary tuning
CHORUS_DURATION_SINGLE = 30.0
CHORUS_DURATION = 10.0
TUNE_SCOPE = 6.0
TUNE_WINDOW = 6.0


# ssm target generation (label index)
DATASET_LABEL_SET = USING_DATASET.getLabels()
SSM_SEMANTIC_LABEL_DIC = defaultdict(int, USING_DATASET.semanticLabelDic())
assert SSM_SEMANTIC_LABEL_DIC["background"] == 0
# melody target generation (label index)
MEL_SEMANTIC_LABEL_DIC = defaultdict(
    int,
    {
        "background": 0,
        "chorus": 1,
        "verse": 2,
    },
)

# Chorus classifier
CLF_TARGET_LABEL = "chorus"
CLF_NON_TARGET_LABEL = "others"
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
CC_RECALL = 0.0
MINIMUM_CHORUS_DUR = 10

# classifier parameters
RD_FOREST_ESTIMATORS = 1000
RD_FOREST_RANDOM_STATE = 42
