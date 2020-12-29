import os
import numpy as np
from copy import copy
from collections import defaultdict


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
PITCH_CHROMA_CLASS = 12
PITCH_CHROMA_COUNT = 100
PITCH_CHROMA_HOP = 1
# PITCH_CHROMA_CLASS = int(os.getenv("PC_CLASS"))
# PITCH_CHROMA_COUNT = int(os.getenv("PC_STEP"))
# PITCH_CHROMA_HOP = int(os.getenv("PC_SEQ"))

# MsafAlgos wrapper
SSM_LOG_THRESH = -4.5

# sequence recurrence algorithm
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
# classifier parameters
RD_FOREST_ESTIMATORS = 1000
RD_FOREST_RANDOM_STATE = 42
# clique class target generation
CC_PRECISION = 0.5
CC_RECALL = 0.0
MINIMUM_CHORUS_DUR = 10


# predict model file
USE_DATASET_NAME = ["RWC_Popular_Dataset", "CCM_Dataset"][0]
USE_MODEL_DIC = {
    algo: f"data/models/{USE_DATASET_NAME}_{algo}_TRAIN.pkl"
    for algo in [
        "scluster",
        "cnmf",
        "sf",
        "olda",
        "foote",
        "gtBoundary",
    ]
}
USE_MODEL_DIC[
    "seqRecur"
] = f"data/models/{USE_DATASET_NAME}_tf{SSM_TRANSFORM_IDENTIFIER}_seqRecur_TRAIN.pkl"
