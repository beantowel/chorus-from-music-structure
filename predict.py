import librosa
import click
import json
import os
import numpy as np
from tqdm import tqdm

from configs.configs import *
from configs.modelConfigs import *
from utility.transform import *
from models.classifier import *
from utility.dataset import DummyDataset
from utility.algorithmsWrapper import AlgoSeqRecur, AlgoSeqRecurSingle
from feature import buildPreprocessDataset


def calculateFeature(audiofile):
    tf = ExtractMel()
    feature = tf.preprocessor(audiofile)
    mels_f = feature['times'], feature['pitches']

    tf = GenerateSSM()
    feature = tf.preprocessor(audiofile)
    ssm_f = feature['times'], tf.logSSM(feature['ssm'][0])
    return ssm_f, mels_f


def writeJsonMetadata(audiofile, mirexFmt, output):
    annotation = []
    for intv, label in zip(*mirexFmt):
        annotation.append({
            'begin': float('%.2f' % intv[0]),
            'end': float('%.2f' % intv[1]),
            'label': label,
        })
    meta = {
        'audio': audiofile,
        'annotation': annotation,
    }
    with open(output, 'w') as f:
        json.dump(meta, f)
    print(f'metadata written to {output}')


def writeMirexOutput(mirexFmt, output):
    intervals, labels = mirexFmt
    mirexOutput = np.array([(x[0], x[1], y)
                            for x, y in zip(intervals, labels)], np.dtype('f, f, U16'))
    np.savetxt(output, mirexOutput, fmt=[
               '%.2f', '%.2f', '%s'], delimiter='\t')
    print(f'mirex format music structure written to {output}')


@click.command()
@click.argument('audiofile', nargs=1, type=click.Path(exists=True))
@click.argument('outputdir', nargs=1, default=PRED_DIR, type=click.Path())
@click.argument('metaOutputdir', nargs=1, default=VIEWER_DATA_DIR, type=click.Path())
@click.option('--algo', nargs=1, type=click.Choice(['multi', 'single']), default='multi')
def main(audiofile, outputdir, metaoutputdir, algo):
    audioFileName = os.path.splitext(os.path.split(audiofile)[-1])[0]
    output = os.path.join(outputdir, audioFileName + '.txt')
    metaOutput = os.path.join(metaoutputdir, audioFileName + '_meta.json')

    ddataset = DummyDataset([audiofile])
    for tf in [ExtractMel(), GenerateSSM(dataset=ddataset), ExtractCliques(dataset=ddataset)]:
        preDataset = Preprocess_Dataset(
            tf.identifier, ddataset)
        preDataset.build(tf.preprocessor, force=False)
    if algo == 'multi':
        algo = AlgoSeqRecur()
    elif algo == 'single':
        algo = AlgoSeqRecurSingle()
    mirexFmt = algo(ddataset, 0)

    writeMirexOutput(mirexFmt, output)
    writeJsonMetadata(audiofile, mergeIntervals(mirexFmt), metaOutput)


if __name__ == '__main__':
    main()
