import librosa
import click
import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from configs.configs import *
from configs.modelConfigs import *
from utility.transform import *
from utility.dataset import DummyDataset
from utility.algorithmsWrapper import AlgoSeqRecur, AlgoSeqRecurSingle
from utility.common import logSSM
from models.classifier import *
from feature import buildPreprocessDataset


def plotMats(matrices, titles, show=SHOW):
    print(f"plot mats[{len(matrices)}]:")
    if len(matrices) > 3:
        fig, axis = plt.subplots(2, (len(matrices) + 1) // 2)
    else:
        fig, axis = plt.subplots(1, len(matrices))
    axis = axis.flatten()
    for i, mat in enumerate(matrices):
        print(f"{titles[i]}{mat.shape}, min:{np.min(mat)}, max:{np.max(mat)}")
        ax = axis[i]
        ax.set_title(titles[i])
        extent = [-1, len(mat) * SSM_TIME_STEP]
        im = ax.imshow(mat, interpolation="none", extent=extent + extent[::-1])
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
    print(f"metadata written to {output}")


def writeMirexOutput(mirexFmt, output):
    intervals, labels = mirexFmt
    mirexOutput = np.array(
        [(x[0], x[1], y) for x, y in zip(intervals, labels)], np.dtype("f, f, U16")
    )
    np.savetxt(output, mirexOutput, fmt=["%.2f", "%.2f", "%s"], delimiter="\t")
    print(f"mirex format music structure written to {output}")


@click.command()
@click.argument("audiofiles", nargs=-1, type=click.Path(exists=True))
@click.option("--outputdir", nargs=1, default=PRED_DIR, type=click.Path())
@click.option("--metaOutputdir", nargs=1, default=VIEWER_DATA_DIR, type=click.Path())
@click.option(
    "--algo", nargs=1, type=click.Choice(["multi", "single"]), default="multi"
)
def main(audiofiles, outputdir, metaoutputdir, algo):
    for audiofile in audiofiles:
        audiofile = os.path.abspath(audiofile)
        audioFileName = os.path.splitext(os.path.split(audiofile)[-1])[0]
        output = os.path.join(outputdir, audioFileName + ".txt")
        metaOutput = os.path.join(metaoutputdir, audioFileName + "_meta.json")

        ddataset = DummyDataset([audiofile])
        for tf in [
            ExtractMel(),
            GenerateSSM(dataset=ddataset),
            ExtractCliques(dataset=ddataset),
        ]:
            preDataset = Preprocess_Dataset(tf.identifier, ddataset)
            preDataset.build(tf.preprocessor, force=False)
        if algo == "multi":
            algo = AlgoSeqRecur()
        elif algo == "single":
            algo = AlgoSeqRecurSingle()
        mirexFmt = algo(ddataset, 0)
        writeMirexOutput(mirexFmt, output)

        # write viewer metadata
        ssm_f, _ = getFeatures(ddataset, 0)
        tf = ExtractCliques(dataset=ddataset)
        origCliques = Preprocess_Dataset(
            tf.identifier, ddataset, transform=tf.transform
        )[0]["cliques"]
        olssm = getLabeledSSM(origCliques, ssm_f[1].shape[-1])
        mats = np.array([ssm_f[1], olssm])
        titles = ["fused SSM", "low level structure"]
        plotMats(mats, titles, show=False)

        figurePath = os.path.join(os.getcwd(), f"data/test/predict_{audioFileName}.svg")
        plt.savefig(figurePath, bbox_inches="tight")
        writeJsonMetadata(audiofile, mergeIntervals(mirexFmt), figurePath, metaOutput)


if __name__ == "__main__":
    main()
