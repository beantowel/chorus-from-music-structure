import os

SA_HOME = os.environ["HOME"] + "/FDU/MIR"
# dependency location and location to store temporary files running the dependencies
ALGO_BASE_DIRS = {
    "JDC": f"{SA_HOME}/Projects/melodyExtraction_JDC",
    "PopMusicHighlighter": f"{SA_HOME}/Projects/results_of_highlighter_on_RWC",
    "TmpDir": "/tmp/MIR",
}
# dataset location and preprocess cache files location
DATASET_BASE_DIRS = {
    "SALAMI": f"{SA_HOME}/dataset/salami",
    "RWC": f"{SA_HOME}/dataset/melodyExtraction/RWC",
    "LocalTemporary_Dataset": f"{SA_HOME}/dataset/localTmp",
}
# output data location
EVAL_RESULT_DIR = "data/evalResult/"
MODELS_DIR = "data/models"
VIEWER_DATA_DIR = "data/viewerMetadata"
PRED_DIR = "data/predict"
# make these directories
mk_dirs = [
    EVAL_RESULT_DIR,
    MODELS_DIR,
    VIEWER_DATA_DIR,
    PRED_DIR,
    ALGO_BASE_DIRS["TmpDir"],
    DATASET_BASE_DIRS["LocalTemporary_Dataset"],
    os.path.join(DATASET_BASE_DIRS["SALAMI"], "features"),
    os.path.join(DATASET_BASE_DIRS["RWC"], "RWC-MDB-P-2001/features"),
]
for path in mk_dirs:
    if not os.path.exists(path):
        os.mkdir(path)

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
    ["ovlp-F", "sovl-F"],
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

SHOW = True if os.getenv("TEST_SHOW") is not None else False
