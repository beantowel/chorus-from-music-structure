import librosa
import scipy
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from utility.GraphDitty.CSMSSMTools import getCSM, getCSMCosine, getCSMCosine_ShiftInvariant
from utility.GraphDitty.SimilarityFusion import doSimilarityFusionWs, getW
from configs.modelConfigs import *


def selfSimilarityMatrix(wavfile, mel=None, win_fac=10, wins_per_block=20, K=5, sr=22050, hop_length=512):
    print(f'loading:{wavfile}')
    y, sr = librosa.load(wavfile, sr=sr)
    nHops = (y.size - hop_length*(win_fac-1)) / hop_length
    intervals = np.arange(0, nHops + 1e-6, win_fac).astype(int)
    print(
        f'{nHops}=({y.size} - {hop_length}*({win_fac}-1))/{hop_length} intvs:{intervals[-1]}')
    # chorma
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=hop_length, bins_per_octave=12*3)
    # mfcc
    S = librosa.feature.melspectrogram(
        y, sr=sr, n_mels=128, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
    lifterexp = 0.6
    coeffs = np.arange(mfcc.shape[0])**lifterexp
    coeffs[0] = 1
    mfcc = coeffs[:, None]*mfcc
    # tempogram
    SUPERFLUX_SIZE = 5
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length,
                                        max_size=SUPERFLUX_SIZE)
    tempogram = librosa.feature.tempogram(
        onset_envelope=oenv, sr=sr, hop_length=hop_length)

    n_frames = np.min([chroma.shape[1], mfcc.shape[1], tempogram.shape[1]])
    intervals = librosa.util.fix_frames(intervals, x_min=0, x_max=n_frames)
    times = intervals * float(hop_length) / float(sr)
    size = n_frames // win_fac
    print(
        f'fixed intervals:{intervals[-1]} hop:{intervals[1]-intervals[0]} size:{size}')
    printArray(mfcc, 'mfcc')
    WMfcc = feature2W(mfcc, size, None, getCSM)
    printArray(chroma, 'chorma')
    WChroma = feature2W(chroma, size, np.median,
                        getCSMCosine_ShiftInvariant(wins_per_block))
    printArray(tempogram, 'tempo')
    WTempo = feature2W(tempogram, size, None, getCSM)

    # melody
    if mel is not None:
        _, pitches = mel
        pitches = pitchChroma(pitches)
        WPitches = feature2W(pitches, size, np.mean,
                             getCSMCosine_ShiftInvariant(wins_per_block))
        Ws = [WMfcc, WChroma, WPitches, WTempo]
    else:
        Ws = [WMfcc, WChroma, WTempo]

    if REC_SMOOTH > 0:
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Ws = [df(W, size=(1, REC_SMOOTH)) for W in Ws]
    W = doSimilarityFusionWs(Ws, K=K, niters=3, reg_diag=1.0, reg_neighbs=0.5)
    res = {
        'Ws': {
            'Fused': W,
            'Melody': WPitches if mel is not None else None,
        },
        'times': times
    }
    return res


def pitchChroma(pitches, n_class=24, n_steps=50):
    ''' input: [t]
        Xinput: [n_steps, t]
        output: [n_class, t]'''
    pitches = deepcopy(pitches)
    size = pitches.shape[-1]
    # set non-voice melody a non-zero value incase negative value in log()
    pitches[pitches <= 0] = 20
    pitches = librosa.hz_to_midi(pitches) * n_class / 12
    pitches = np.remainder(pitches.astype(int), n_class)
    pitches = np.expand_dims(pitches, axis=0)
    assert (pitches < n_class).all()
    XPitches = librosa.feature.stack_memory(
        pitches, n_steps=n_steps, mode='edge')
    assert (XPitches < n_class).all(
    ), 'suprise! just try again the bug might gone'
    res = np.zeros((n_class, size))
    for t in range(size):
        res[:, t] = np.bincount(XPitches[:, t], minlength=n_class)
    return res


def feature2W(feature, size, aggregator, simFunction, wins_per_block=20, K=5):
    length = feature.shape[-1]
    hop = length // (size - 1)
    intervals = np.arange(0, length, hop).astype(int)
    intervals = librosa.util.fix_frames(
        intervals, x_min=0, x_max=hop * (size - 1))
    feature = librosa.util.sync(feature, intervals, aggregate=aggregator)
    Xfeature = librosa.feature.stack_memory(
        feature, n_steps=wins_per_block, mode='edge').T
    Dfeature = simFunction(Xfeature, Xfeature)
    Wfeature = getW(Dfeature, K)
    assert not np.isnan(np.sum(Wfeature)), f'{Dfeature}'
    return Wfeature


def printArray(arr, name, show=False):
    print(f'{name}{arr.shape}:{np.min(arr)}~{np.max(arr)}')
    if show:
        plt.imshow(arr)
        plt.colorbar()
        plt.show()
