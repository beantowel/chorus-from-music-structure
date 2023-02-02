import os
import logging

join_path = lambda x: os.path.join(*x.split("/"))
SA_HOME = os.getcwd()
# dependency location and location to store temporary files running the dependencies
ALGO_BASE_DIRS = {
    "JDC": join_path(f"{os.getcwd()}/third_party/melodyExtraction_JDC"),
    "SSL": join_path(f"{os.getcwd()}/third_party/melodyExtraction_SSL"),
    "PopMusicHighlighter": join_path(f"{os.getcwd()}/third_party/pop-music-highlighter"),
    "TmpDir": join_path(f"{SA_HOME}/tmp"),
}
# dataset location and preprocess cache files location
DATASET_BASE_DIRS = {
    "SALAMI": join_path(f"{SA_HOME}/dataset/salami"),
    "RWC": join_path(f"{SA_HOME}/dataset/RWC"),
    "RWC_accomp": join_path(f"{SA_HOME}/dataset/RWC-accompaniment"),
    "CCM": join_path(f"{SA_HOME}/dataset/CCM_Structure"),
    "Huawei": join_path(f"{SA_HOME}/dataset/Huawei"),
    "LocalTemporary_Dataset": join_path(f"{SA_HOME}/dataset/localTmp"),
}
# output data location
EVAL_RESULT_DIR = "data/evalResult/"
MODELS_DIR = "data/models"
VIEWER_DATA_DIR = "data/viewerMetadata"
PRED_DIR = "data/predict"

# evaluation settings
FORCE_EVAL = False
METRIC_NAMES = [
    "ovlp-P",
    "ovlp-R",
    "ovlp-F",
    "sovl-P",
    "sovl-R",
    "sovl-F",
    "dtct-P",
    "dtct-R",
    "dtct-F",
]
PLOT_METRIC_FIELDS = [
    ["ovlp-F"],
    ["sovl-F"],
    # ["dtct-P"],
    # ["dtct-R"],
    # ["dtct-F"],
]
# PLOT_METRIC_FIELDS = [
#     ['ovlp-P', 'ovlp-R', 'ovlp-F'],
#     ['sovl-P', 'sovl-R', 'sovl-F'],
# ]
# PLOT_METRIC_FIELDS = [
#     ['ovlp-P', 'ovlp-R'],
#     ['ovlp-F', 'sovl-F'],
# ]
DETECTION_WINDOW = 3

# logging settings
DEBUG = True if os.getenv("DEBUG") is not None else False
logger = logging.getLogger("chorus_detector")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG) if DEBUG else ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# make these directories
mk_dirs = [
    EVAL_RESULT_DIR,
    MODELS_DIR,
    VIEWER_DATA_DIR,
    PRED_DIR,
    ALGO_BASE_DIRS["TmpDir"],
    DATASET_BASE_DIRS["LocalTemporary_Dataset"],
]
for path in mk_dirs:
    if not os.path.exists(path):
        dirname = os.path.dirname(path)
        if os.path.exists(dirname):
            os.mkdir(path)
        else:
            logger.warn(f"directory={dirname} does not exist")

# process numbers for parallel computing
NUM_WORKERS = os.cpu_count() // 2 if not DEBUG else 1
