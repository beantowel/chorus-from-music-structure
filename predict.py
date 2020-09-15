import librosa
import click
import json
import os
import string
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from configs.configs import logger, DEBUG, PRED_DIR, VIEWER_DATA_DIR, NUM_WORKERS
from configs.modelConfigs import SSM_TIME_STEP, CLF_TARGET_LABEL
from utility.transform import ExtractMel, GenerateSSM, ExtractCliques, getFeatures
from utility.dataset import DummyDataset, Preprocess_Dataset
from utility.algorithmsWrapper import AlgoSeqRecur, AlgoSeqRecurSingle
from utility.common import logSSM, extractFunctions, getLabeledSSM, mergeIntervals

# from models.classifier import *
from feature import buildPreprocessDataset


def plotMats(matrices, titles, show=DEBUG):
    logger.debug(f"plot mats[{len(matrices)}]:")
    if len(matrices) > 3:
        _, axis = plt.subplots(2, (len(matrices) + 1) // 2)
    else:
        _, axis = plt.subplots(1, len(matrices))
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


@click.command()
@click.argument("audiofiles", nargs=-1, type=click.Path(exists=True))
@click.option("--outputdir", nargs=1, default=PRED_DIR, type=click.Path())
@click.option("--metaOutputdir", nargs=1, default=VIEWER_DATA_DIR, type=click.Path())
@click.option(
    "--algo", nargs=1, type=click.Choice(["multi", "single"]), default="multi"
)
@click.option("--force", nargs=1, type=click.BOOL, default=False)
@click.option("--workers", nargs=1, type=click.INT, default=NUM_WORKERS)
def main(audiofiles, outputdir, metaoutputdir, algo, force, workers):
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

    for i, pair in enumerate(ddataset.pathPairs):
        audioFileName, audiofile, _ = pair
        audiofile = os.path.abspath(audiofile)
        output = os.path.join(outputdir, audioFileName + ".txt")
        metaOutput = os.path.join(metaoutputdir, audioFileName + "_meta.json")

        logger.info(f'processing "{audiofile}"')
        if algo == "multi":
            algo = AlgoSeqRecur()
        elif algo == "single":
            algo = AlgoSeqRecurSingle()
        mirexFmt = algo(ddataset, i)
        writeMirexOutput(mirexFmt, output)

        # write viewer metadata
        ssm_f, _ = getFeatures(ddataset, i)
        tf = ExtractCliques(dataset=ddataset)
        origCliques = Preprocess_Dataset(
            tf.identifier, ddataset, transform=tf.transform
        )[i]["cliques"]
        cliques = algo._process(ddataset, i, ssm_f)

        olssm = getLabeledSSM(origCliques, ssm_f[1].shape[-1])
        lssm = getLabeledSSM(cliques, ssm_f[1].shape[-1])
        olssm = drawSegments(mirexFmt, mirexFmt, olssm, ssm_f[0])
        mats = np.array([ssm_f[1], lssm, olssm])
        titles = ["fused SSM", "result structure", "low level structure"]
        plotMats(mats, titles, show=False)

        figurePath = os.path.join(os.getcwd(), f"data/test/predict_{audioFileName}.svg")
        plt.savefig(figurePath, bbox_inches="tight")
        writeJsonMetadata(audiofile, mergeIntervals(mirexFmt), figurePath, metaOutput)
        if DEBUG:
            plt.show()


if __name__ == "__main__":
    main()
