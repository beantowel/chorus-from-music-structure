import os
import logging

SA_HOME = os.environ["HOME"] + "/FDU/MIR"
# dependency location and location to store temporary files running the dependencies
ALGO_BASE_DIRS = {
    "JDC": f"{os.getcwd()}/third_party/melodyExtraction_JDC",
    "PopMusicHighlighter": f"{os.getcwd()}/third_party/pop-music-highlighter",
    "TmpDir": "/tmp/MIR",
}
# dataset location and preprocess cache files location
DATASET_BASE_DIRS = {
    "SALAMI": f"{SA_HOME}/dataset/salami",
    "RWC": f"{SA_HOME}/dataset/RWC",
    "RWC_accomp": f"{SA_HOME}/dataset/RWC-accompaniment",
    "CCM": f"{SA_HOME}/dataset/CCM_Structure",
    "LocalTemporary_Dataset": f"{SA_HOME}/dataset/localTmp",
}
# output data location
EVAL_RESULT_DIR = "data/evalResult/"
MODELS_DIR = "data/models"
VIEWER_DATA_DIR = "data/viewerMetadata"
PRED_DIR = "data/predict"

# process numbers for parallel computing
NUM_WORKERS = os.cpu_count() // 2

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
    # os.path.join(DATASET_BASE_DIRS["SALAMI"], "features"),
    os.path.join(DATASET_BASE_DIRS["RWC"], "RWC-MDB-P-2001/features"),
    os.path.join(DATASET_BASE_DIRS["CCM"], "features"),
]
for path in mk_dirs:
    if not os.path.exists(path):
        dirname = os.path.dirname(path)
        if os.path.exists(dirname):
            os.mkdir(path)
        else:
            logger.warn(f"directory={dirname} does not exist")
