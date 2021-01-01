from collections import defaultdict

from utility.dataset import (
    RWC_Popular_Dataset,
    RWC_Popular_Dataset_accomp,
    SALAMI_Dataset,
    CCM_Dataset,
    Huawei_Dataset,
)
from configs.modelConfigs import SSM_TRANSFORM_IDENTIFIER, USE_DATASET_NAME

DATASET_DIC = {
    "RWC_Popular_Dataset": RWC_Popular_Dataset(),
    "RWC_Popular_Dataset_accomp": RWC_Popular_Dataset_accomp(),
    "CCM_Dataset": CCM_Dataset(),
    "Huawei_Dataset": Huawei_Dataset(),
}
USING_DATASET = DATASET_DIC[USE_DATASET_NAME]

# ssm target generation (label index)
CLF_SPLIT_RATIO = 0.8
RANDOM_SEED = 114514
CLF_TRAIN_SET, CLF_VAL_SET = USING_DATASET.randomSplit(
    CLF_SPLIT_RATIO, seed=RANDOM_SEED
)
CHORUS_CLASSIFIER_TRAIN_DATA_FILE = {
    algo: f"data/models/{USING_DATASET.__class__.__name__}_{algo}_TRAIN.pkl"
    for algo in [
        "scluster",
        "cnmf",
        "sf",
        "olda",
        "foote",
        "gtBoundary",
    ]
}
CHORUS_CLASSIFIER_TRAIN_DATA_FILE[
    "seqRecur"
] = f"data/models/{USING_DATASET.__class__.__name__}_tf{SSM_TRANSFORM_IDENTIFIER}_seqRecur_TRAIN.pkl"

CHORUS_CLASSIFIER_VAL_DATA_FILE = {
    key: val.replace("TRAIN.pkl", "VAL.pkl")
    for key, val in CHORUS_CLASSIFIER_TRAIN_DATA_FILE.items()
}
