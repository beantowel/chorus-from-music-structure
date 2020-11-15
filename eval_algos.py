import click
import numpy as np
import pandas as pd

from utility.dataset import RWC_Popular_Dataset, CCM_Dataset, RWC_Popular_Dataset_accomp
from utility.algorithmsWrapper import (
    AlgoSeqRecur,
    AlgoSeqRecurBound,
    AlgoSeqRecurSingle,
    GroudTruthStructure,
    MsafAlgos,
    MsafAlgosBdryOnly,
    MsafAlgosBound,
    PopMusicHighlighter,
    RefraiD,
)
from utility.metrics import AlgoEvaluator, Metrics_Saver
from configs.configs import EVAL_RESULT_DIR, FORCE_EVAL, METRIC_NAMES, logger
from configs.modelConfigs import (
    CLF_VAL_SET,
    CLF_TRAIN_SET,
    USING_DATASET,
    CLF_SPLIT_RATIO,
    RANDOM_SEED,
)

loaders = {
    "RWC_Popular": RWC_Popular_Dataset(),
    "CCM": CCM_Dataset(),
    "RWC_Popular_accomp": RWC_Popular_Dataset_accomp(),
    # 'SALAMI_functions': SALAMI_Dataset(annotation='functions'),
}
algos = {
    "seqRecur": AlgoSeqRecur(),
    "seqRecur+": AlgoSeqRecurBound(),
    "seqRecurS": AlgoSeqRecurSingle(),
    "highlighter": PopMusicHighlighter(),
    "refraiD": RefraiD(),
    "scluster": MsafAlgos("scluster"),
    "sf": MsafAlgosBdryOnly("sf"),
    "olda": MsafAlgosBdryOnly("olda"),
    "foote": MsafAlgosBdryOnly("foote"),
    "scluster+": MsafAlgosBound("scluster"),
    "sf+": MsafAlgosBound("sf"),
    "olda+": MsafAlgosBound("olda"),
    "cnmf": MsafAlgos("cnmf"),
    "cnmf+": MsafAlgosBound("cnmf"),
    "foote+": MsafAlgosBound("foote"),
    "gtBoundary": GroudTruthStructure(),
}
algo_order = [
    "seqRecur",
    "seqRecurS",
    "olda",
    "highlighter",
    # "refraiD",
    "scluster",
    "sf",
    "cnmf",
    "foote",
    "gtBoundary",
]
# algo_order += [
#     "seqRecur+",
#     "olda+",
#     "scluster+",
#     "sf+",
#     "cnmf+",
#     "foote+",
# ]


def printResult(aName, metrics):
    logger.info(f"average result, algoName={aName}:")
    logger.info(f"metricNames={METRIC_NAMES}")
    logger.info(f"metric={np.mean(metrics, axis=0)}")


def dumpResult(saver):
    saver.writeFullResults(EVAL_RESULT_DIR)
    saver.writeAveResults(EVAL_RESULT_DIR)
    saver.saveViolinPlot(EVAL_RESULT_DIR, order=algo_order)


def updateViews(evalAlgos, dName):
    train, val = loaders[dName].randomSplit(CLF_SPLIT_RATIO, seed=RANDOM_SEED)
    loader_views = {
        "_VAL": val,
        "_TRAIN": train,
    }
    for vName, vLoader in loader_views.items():
        for dName, dLoader in loaders.items():
            if vLoader.__class__ is dLoader.__class__:
                name = dName + vName
                logger.info(
                    "----------------------------------------------------------"
                )
                logger.info(f"loader view, name={name}")
                viewSaver = Metrics_Saver(name)
                dSaver = Metrics_Saver(dName)
                dSaver.load(EVAL_RESULT_DIR)
                for aName in dSaver.algoNames:
                    titles = [pp.title for pp in vLoader.pathPairs]
                    metrics, _ = dSaver.getResult(aName, titles)
                    viewSaver.addResult(aName, metrics, titles)
                    if aName in evalAlgos:
                        printResult(aName, metrics)
                dumpResult(viewSaver)


def findLoader(clz):
    for dName, dLoader in loaders.items():
        if clz is dLoader.__class__:
            return {dName: dLoader}
    raise KeyError


@click.command()
@click.option(
    "--force", default=FORCE_EVAL, type=click.BOOL, help="overwrite evaluation results"
)
@click.option(
    "--dataset", default=None, type=click.STRING, help="using specific dataset"
)
@click.option(
    "--algorithm", default=None, type=click.STRING, help="using specific algorithm"
)
def main(force, dataset, algorithm):
    if dataset is None:
        evalLoader = loaders
    elif dataset == "auto":
        evalLoader = findLoader(USING_DATASET.__class__)
    else:
        evalLoader = {dataset: loaders[dataset]}
    if algorithm is None:
        evalAlgos = algos
    else:
        evalAlgos = {algorithm: algos[algorithm]}

    for dName, loader in evalLoader.items():
        logger.info("-----------------------eval_algos---------------------------")
        logger.info(f"processing datasetloader, name={dName}")
        saver = Metrics_Saver(dName)
        # run incremental evaluation by default
        saver.load(EVAL_RESULT_DIR)
        for aName, algo in evalAlgos.items():
            # avoid duplicate evaluation
            if (aName not in saver.algoNames) or force:
                if force and (aName in saver.algoNames):
                    logger.info(f"re-eval algo, name={aName}")
                else:
                    logger.info(f"algo, name={aName}")

                if hasattr(algo, "clf"):
                    algo.clf.train()
                ae = AlgoEvaluator(loader, algo)
                metrics, titles = ae()
                printResult(aName, metrics)

                if force and (aName in saver.algoNames):
                    saver.reWriteResult(aName, metrics, titles)
                else:
                    saver.addResult(aName, metrics, titles)
                # save result every iter
                saver.dump(EVAL_RESULT_DIR)
            else:
                logger.info(f"!! skipping algo, name={aName}")
        dumpResult(saver)
        updateViews(evalAlgos, dName)


if __name__ == "__main__":
    main()
