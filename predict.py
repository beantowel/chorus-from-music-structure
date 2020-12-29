import librosa
import click
import json
import os
import string
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.classifier import ChorusClassifier, chorusDetection, getFeatures
from models.pickSingle import maxOverlap, tuneIntervals
from configs.configs import logger, DEBUG, PRED_DIR, VIEWER_DATA_DIR, NUM_WORKERS
from configs.modelConfigs import (
    SSM_TIME_STEP,
    CLF_TARGET_LABEL,
    USE_MODEL_DIC,
    CHORUS_DURATION,
    CHORUS_DURATION_SINGLE,
    TUNE_WINDOW,
)
from utility.transform import ExtractMel, GenerateSSM, ExtractCliques, getFeatures
from utility.dataset import DummyDataset, Preprocess_Dataset, buildPreprocessDataset
from utility.algorithmsWrapper import (
    AlgoSeqRecur,
    AlgoSeqRecurSingle,
    PopMusicHighlighter,
    AlgoMixed,
    MsafAlgosBdryOnly,
)
from utility.common import logSSM, extractFunctions, getLabeledSSM, mergeIntervals


def plotMats(matrices, titles, show=DEBUG):
    logger.debug(f"plot mats[{len(matrices)}]:")
    if len(matrices) > 3:
        _, axis = plt.subplots(2, (len(matrices) + 1) // 2)
    else:
        _, axis = plt.subplots(1, len(matrices))
    if len(matrices) == 1:
        axis = np.array([axis])
    axis = axis.flatten()
    for i, mat in enumerate(matrices):
        logger.debug(f"{titles[i]}{mat.shape}, min={np.min(mat)}, max={np.max(mat)}")
        ax = axis[i]
        ax.set_title(f"({string.ascii_lowercase[i]}) {titles[i]}")
        extent = [-1, len(mat) * SSM_TIME_STEP]
        ax.imshow(mat, interpolation="none", extent=extent + extent[::-1])
        ax.set_xlabel("time/s")
        # fig.colorbar(im, orientation=orien, ax=ax)
    plt.tight_layout()
    if show:
        plt.show()


def writeJsonMetadata(audiofile, predicted, figure, output, gt=None):
    def annotation(mirexFmt):
        annotation = []
        for intv, label in zip(*mirexFmt):
            annotation.append(
                {
                    "begin": float("%.2f" % intv[0]),
                    "end": float("%.2f" % intv[1]),
                    "label": label,
                }
            )
        return annotation

    meta = {
        "audio": audiofile,
        "annotation": annotation(predicted),
        "gt_annotation": annotation(gt) if gt is not None else None,
        "figure": figure,
    }
    with open(output, "w") as f:
        json.dump(meta, f)
    logger.info(f"metadata written to {output}")


def writeMirexOutput(mirexFmt, output):
    intervals, labels = mirexFmt
    mirexOutput = np.array(
        [(x[0], x[1], y) for x, y in zip(intervals, labels)], np.dtype("f, f, U16")
    )
    np.savetxt(output, mirexOutput, fmt=["%.2f", "%.2f", "%s"], delimiter="\t")
    logger.info(f"mirex format music structure written to {output}")


def drawSegments(ref, est, ssm, times):
    def drawIntervals(mirexFmt, ssm, up):
        size = ssm.shape[-1]
        intvs, labels = mirexFmt
        labels = extractFunctions(labels)
        labelClz = np.array([1 if label == CLF_TARGET_LABEL else 0 for label in labels])
        intvs = (intvs / times[-1] * size).astype(int)
        for intv, label in zip(intvs, labelClz):
            if up:
                ssm[: size // 2, intv[0] : intv[1]] += label
            else:
                ssm[size // 2 :, intv[0] : intv[1]] += label
        return ssm

    ssm = ssm / np.max(np.abs(ssm))
    ssm = drawIntervals(ref, ssm, True)
    ssm = drawIntervals(est, ssm, False)
    return ssm


def switchPred(algo):
    if algo == "highlighter":
        return PopMusicHighlighter()
    elif algo == "sf":
        return MsafAlgosBdryOnly("sf", USE_MODEL_DIC["sf"])
    elif algo == "mixed":
        return AlgoMixed(USE_MODEL_DIC["seqRecur"])
    else:
        return None


@click.command()
@click.argument("audiofiles", nargs=-1, type=click.Path(exists=True))
@click.option("--outputdir", nargs=1, default=PRED_DIR, type=click.Path())
@click.option("--metaOutputdir", nargs=1, default=VIEWER_DATA_DIR, type=click.Path())
@click.option(
    "--algo",
    nargs=1,
    type=click.Choice(["multi", "single", "highlighter", "sf", "mixed"]),
    default="multi",
)
@click.option(
    "--force", nargs=1, type=click.BOOL, default=True, help="overwrite cached features."
)
@click.option("--workers", nargs=1, type=click.INT, default=NUM_WORKERS)
def main(audiofiles, outputdir, metaoutputdir, algo, force, workers):
    logger.debug(f"algo={algo}")
    logger.info(f"preprocess to generate features")
    ddataset = DummyDataset(audiofiles)
    transforms = [
        ExtractMel(),
        GenerateSSM(dataset=ddataset),
        ExtractCliques(dataset=ddataset),
    ]
    for tf in transforms:
        preDataset = Preprocess_Dataset(tf.identifier, ddataset)
        preDataset.build(tf.preprocessor, force=force, num_workers=workers)

    predictor = AlgoSeqRecur(trainFile=USE_MODEL_DIC["seqRecur"])
    predictor2 = switchPred(algo)
    for i, pair in enumerate(ddataset.pathPairs):
        audioFileName, audiofile, _ = pair
        audiofile = os.path.abspath(audiofile)
        output = os.path.join(outputdir, audioFileName + ".txt")
        metaOutput = os.path.join(metaoutputdir, audioFileName + "_meta.json")

        ssm_f, mels_f = getFeatures(ddataset, i)
        cliques = predictor._process(ddataset, i, ssm_f)
        mirexFmt = chorusDetection(cliques, ssm_f[0], mels_f, predictor.clf)
        if algo == "multi":
            mirexFmt = tuneIntervals(
                mirexFmt, mels_f, chorusDur=CHORUS_DURATION, window=TUNE_WINDOW
            )
        elif algo == "single":
            mirexFmtSingle = maxOverlap(
                mirexFmt, chorusDur=CHORUS_DURATION_SINGLE, centering=False
            )
            mirexFmtSingle = tuneIntervals(
                mirexFmtSingle,
                mels_f,
                chorusDur=CHORUS_DURATION_SINGLE,
                window=TUNE_WINDOW,
            )

        # plot mats
        tf = ExtractCliques(dataset=ddataset)
        origCliques = Preprocess_Dataset(
            tf.identifier, ddataset, transform=tf.transform
        )[i]["cliques"]
        olssm = getLabeledSSM(origCliques, ssm_f[1].shape[-1])
        lssm = getLabeledSSM(cliques, ssm_f[1].shape[-1])
        olssm = drawSegments(mirexFmt, mirexFmt, olssm, ssm_f[0])
        mats = np.array([ssm_f[1], lssm, olssm])
        titles = ["fused SSM", "result structure", "low level structure"]
        plotMats(mats, titles, show=False)

        # write output and viewer metadata
        if algo == "single":
            mirexFmt = mirexFmtSingle
        elif algo not in ["single", "multi"]:
            mirexFmt = predictor2(ddataset, i)

        writeMirexOutput(mirexFmt, output)
        figurePath = os.path.join(os.getcwd(), f"data/test/predict_{audioFileName}.svg")
        plt.savefig(figurePath, bbox_inches="tight")
        writeJsonMetadata(audiofile, mergeIntervals(mirexFmt), figurePath, metaOutput)
        if DEBUG:
            plt.show()


if __name__ == "__main__":
    main()
