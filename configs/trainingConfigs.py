from collections import defaultdict

from utility.dataset import (
    RWC_Popular_Dataset,
    RWC_Popular_Dataset_accomp,
    SALAMI_Dataset,
    CCM_Dataset,
)
from configs.modelConfigs import SSM_TRANSFORM_IDENTIFIER, USE_DATASET_NAME

USING_DATASET = RWC_Popular_Dataset()
USE_DATASET_DIC = {
    "RWC": RWC_Popular_Dataset,
    "CCMusic": CCM_Dataset,
}
USING_DATASET = USE_DATASET_DIC[USE_DATASET_NAME]()

# ssm target generation (label index)
CLF_SPLIT_RATIO = 0.8
RANDOM_SEED = 114514
CLF_TRAIN_SET, CLF_VAL_SET = USING_DATASET.randomSplit(
    CLF_SPLIT_RATIO, seed=RANDOM_SEED
)
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
