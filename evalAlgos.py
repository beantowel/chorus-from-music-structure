import click
import numpy as np
import pandas as pd

from utility.dataset import *
from utility.transform import *
from utility.algorithmsWrapper import *
from utility.metrics import AlgoEvaluator, Metrics_Saver
from configs.configs import EVAL_RESULT_DIR, FORCE_EVAL, METRIC_NAMES
from configs.modelConfigs import CLF_VAL_SET, CLF_TRAIN_SET, USING_DATASET

loaders = {
    'RWC_Popular': RWC_Popular_Dataset(),
    # 'SALAMI_functions': SALAMI_Dataset(annotation='functions'),
}
algos = {
    'seqRecur': AlgoSeqRecur(),
    'seqRecur+': AlgoSeqRecurBound(),
    'seqRecurS': AlgoSeqRecurSingle(),
    'highlighter': PopMusicHighlighter(),
    'scluster': MsafAlgos('scluster'),
    'sf': MsafAlgosBdryOnly('sf'),
    'olda': MsafAlgosBdryOnly('olda'),
    'foote': MsafAlgosBdryOnly('foote'),
    'scluster+': MsafAlgosBound('scluster'),
    'sf+': MsafAlgosBound('sf'),
    'olda+': MsafAlgosBound('olda'),
    'cnmf': MsafAlgos('cnmf'),
    'cnmf+': MsafAlgosBound('cnmf'),
    'foote+': MsafAlgosBound('foote'),
    'gt': GroudTruthStructure(),
    # 'vmo+': MsafAlgosBound('vmo'),
    # 'vmo': MsafAlgos('vmo'),
}
algo_order = ['seqRecur', 'seqRecurS', 'highlighter', 'scluster', 'sf', 'olda', 'cnmf', 'foote'] + [
    'seqRecur+', 'scluster+', 'sf+', 'olda+', 'cnmf+', 'foote+', 'gt']
loader_views = {
    '_VAL': CLF_VAL_SET,
    '_TRAIN': CLF_TRAIN_SET,
}


def updateViews(evalAlgos):
    for vName, vLoader in loader_views.items():
        for dName, dLoader in loaders.items():
            if vLoader.__class__ is dLoader.__class__:
                name = dName + vName
                print('----------------------------------------------------------')
                print(f'loader view: [{name}]')
                viewSaver = Metrics_Saver(name)
                dSaver = Metrics_Saver(dName)
                dSaver.load(EVAL_RESULT_DIR)
                for aName in dSaver.algoNames:
                    titles = [pp.title for pp in vLoader.pathPairs]
                    metrics, _ = dSaver.getResult(aName, titles)
                    viewSaver.addResult(aName, metrics, titles)
                    if aName in evalAlgos:
                        print(f'[{aName}] average result:')
                        print(METRIC_NAMES)
                        print(np.mean(metrics, axis=0))
                viewSaver.writeFullResults(EVAL_RESULT_DIR)
                viewSaver.writeAveResults(EVAL_RESULT_DIR)
                viewSaver.saveViolinPlot(EVAL_RESULT_DIR, order=algo_order)


def findLoader(clz):
    for dName, dLoader in loaders.items():
        if clz is dLoader.__class__:
            return {dName: dLoader}
    raise KeyError


@click.command()
@click.option('--force', default=FORCE_EVAL, type=click.BOOL, help='overwrite evaluation results')
@click.option('--dataset', default=None, type=click.STRING, help='using specific dataset')
@click.option('--algorithm', default=None, type=click.STRING, help='using specific algorithm')
def main(force, dataset, algorithm):
    if dataset is None:
        evalLoader = loaders
    elif dataset == 'auto':
        evalLoader = findLoader(USING_DATASET.__class__)
    else:
        evalLoader = {dataset: loaders[dataset]}
    if algorithm is None:
        evalAlgos = algos
    else:
        evalAlgos = {algorithm: algos[algorithm]}

    for dName, loader in evalLoader.items():
        print('----------------------------------------------------------')
        print(f'loader: [{dName}]')
        saver = Metrics_Saver(dName)
        # run incremental evaluation by default
        saver.load(EVAL_RESULT_DIR)
        for aName, algo in evalAlgos.items():
            # avoid duplicate evaluation
            if (aName not in saver.algoNames) or force:
                if force and (aName in saver.algoNames):
                    print(f're-eval algo: [{aName}]')
                else:
                    print(f'algo: [{aName}]')

                if hasattr(algo, 'clf'):
                    algo.clf.train()
                ae = AlgoEvaluator(loader, algo)
                metrics, titles = ae()
                print(f'[{aName}] average result:')
                print(METRIC_NAMES)
                print(np.mean(metrics, axis=0))

                if force and (aName in saver.algoNames):
                    saver.reWriteResult(aName, metrics, titles)
                else:
                    saver.addResult(aName, metrics, titles)
                # save result every iter
                saver.dump(EVAL_RESULT_DIR)
            else:
                print(f'! skipping algo: [{aName}]')
        saver.writeFullResults(EVAL_RESULT_DIR)
        saver.writeAveResults(EVAL_RESULT_DIR)
        saver.saveViolinPlot(EVAL_RESULT_DIR, order=algo_order)
    updateViews(evalAlgos)


if __name__ == '__main__':
    main()
