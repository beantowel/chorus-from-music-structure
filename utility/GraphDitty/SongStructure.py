"""
Programmer: Chris Tralie
Purpose: To provide an interface for loading music, computing features, and
doing similarity fusion on those features to make a weighted adjacency matrix
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate
import os
import librosa
import librosa.display
import subprocess

from utility.GraphDitty.CSMSSMTools import getCSM, getCSMCosine, getCSMCosine_ShiftInvariant
from utility.GraphDitty.SimilarityFusion import doSimilarityFusionWs, getW

REC_SMOOTH = 9
MANUAL_AUDIO_LOAD = False
FFMPEG_BINARY = "ffmpeg"


def getFusedSimilarity(filename, sr, hop_length, win_fac, wins_per_block, K, reg_diag, reg_neighbs, niters, do_animation, plot_result, do_crema=True):
    """
    Load in filename, compute features, average/stack delay, and do similarity
    network fusion (SNF) on all feature types
    Parameters
    ----------
    filename: string
        Path to music file
    sr: int
        Sample rate at which to sample file
    hop_length: int
        Hop size between frames in chroma and mfcc
    win_fac: int
        Number of frames to average (i.e. factor by which to downsample)
        If negative, then do beat tracking, and subdivide by |win_fac| times within each beat
    wins_per_block: int
        Number of aggregated windows per sliding window block
    K: int
        Number of nearest neighbors in SNF.  If -1, then autotuned to sqrt(N)
        for an NxN similarity matrix
    reg_diag: float 
        Regularization for self-similarity promotion
    reg_neighbs: float
        Regularization for direct neighbor similarity promotion
    niters: int
        Number of iterations in SNF
    do_animation: boolean
        Whether to plot and save images of the evolution of SNF
    plot_result: boolean
        Whether to plot the result of the fusion
    do_crema: boolean
        Whether to include precomputed crema in the fusion
    Returns
    -------
    {'Ws': An dictionary of weighted adjacency matrices for individual features
                    and the fused adjacency matrix, 
            'times': Time in seconds of each row in the similarity matrices,
            'K': The number of nearest neighbors actually used} 
    """
    # Step 1: Load audio
    print("Loading %s..." % filename)
    if MANUAL_AUDIO_LOAD:
        subprocess.call([FFMPEG_BINARY, "-i", filename, "-ar",
                         "%i" % sr, "-ac", "1", "%s.wav" % filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sr, y = sio.wavfile.read("%s.wav" % filename)
        y = y/2.0**15
        os.remove("%s.wav" % filename)
    else:
        y, sr = librosa.load(filename, sr=sr)

    # Step 2: Figure out intervals to which to sync features
    if win_fac > 0:
        # Compute features in intervals evenly spaced by the hop size
        # but average within "win_fac" intervals of hop_length
        # nHops = int((y.size-hop_length*(win_fac-1))/hop_length)
        nHops = (y.size - hop_length*(win_fac-1)) / hop_length
        intervals = np.arange(0, nHops, win_fac).astype(int)
        print(f'{nHops}={y.size, hop_length, win_fac} intvs:{intervals[-1]}')
    else:
        # Compute features in intervals which are subdivided beats
        # by a factor of |win_fac|
        C = np.abs(librosa.cqt(y=y, sr=sr))
        _, beats = librosa.beat.beat_track(
            y=y, sr=sr, trim=False, start_bpm=240)
        intervals = librosa.util.fix_frames(beats, x_max=C.shape[1])
        intervals = librosa.segment.subsegment(
            C, intervals, n_segments=abs(win_fac))

    # Step 3: Compute features
    # 1) CQT chroma with 3x oversampling in pitch
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=hop_length, bins_per_octave=12*3)

    # 2) Exponentially liftered MFCCs
    S = librosa.feature.melspectrogram(
        y, sr=sr, n_mels=128, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
    lifterexp = 0.6
    coeffs = np.arange(mfcc.shape[0])**lifterexp
    coeffs[0] = 1
    mfcc = coeffs[:, None]*mfcc

    # 3) Tempograms
    #  Use a super-flux max smoothing of 5 frequency bands in the oenv calculation
    SUPERFLUX_SIZE = 5
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length,
                                        max_size=SUPERFLUX_SIZE)
    tempogram = librosa.feature.tempogram(
        onset_envelope=oenv, sr=sr, hop_length=hop_length)

    # 4) Crema
    if do_crema:
        matfilename = "%s_crema.mat" % filename
        if not os.path.exists(matfilename):
            print("****WARNING: PRECOMPUTED CREMA DOES NOT EXIST****")
            do_crema = False
        else:
            data = sio.loadmat(matfilename)
            fac = (float(sr)/44100.0)*4096.0/hop_length
            times_orig = fac*np.arange(len(data['chord_bass']))
            times_new = np.arange(mfcc.shape[1])
            interp = scipy.interpolate.interp1d(
                times_orig, data['chord_pitch'].T, kind='nearest', fill_value='extrapolate')
            chord_pitch = interp(times_new)

    # Step 4: Synchronize features to intervals
    n_frames = np.min([chroma.shape[1], mfcc.shape[1], tempogram.shape[1]])
    if do_crema:
        n_frames = min(n_frames, chord_pitch.shape[1])
    # median-aggregate chroma to suppress transients and passing tones
    intervals = librosa.util.fix_frames(intervals, x_min=0, x_max=n_frames)
    print(
        f'fixed intvs:{intervals[-1]} hop:{win_fac}\nchroma,mfcc,tempo:{chroma.shape, mfcc.shape, tempogram.shape}')
    times = intervals*float(hop_length)/float(sr)

    chroma = librosa.util.sync(chroma, intervals, aggregate=np.median)
    chroma = chroma[:, :n_frames]
    mfcc = librosa.util.sync(mfcc, intervals)
    mfcc = mfcc[:, :n_frames]
    tempogram = librosa.util.sync(tempogram, intervals)
    tempogram = tempogram[:, :n_frames]
    if do_crema:
        chord_pitch = librosa.util.sync(chord_pitch, intervals)
        chord_pitch = chord_pitch[:, :n_frames]

    # Step 5: Do a delay embedding and compute SSMs
    XChroma = librosa.feature.stack_memory(
        chroma, n_steps=wins_per_block, mode='edge').T
    # DChroma = getCSMCosine(XChroma, XChroma)  # Cosine distance
    DChroma = getCSMCosine_ShiftInvariant(wins_per_block)(
        XChroma, XChroma)  # Cosine distance
    XMFCC = librosa.feature.stack_memory(
        mfcc, n_steps=wins_per_block, mode='edge').T
    DMFCC = getCSM(XMFCC, XMFCC)  # Euclidean distance
    XTempogram = librosa.feature.stack_memory(
        tempogram, n_steps=wins_per_block, mode='edge').T
    DTempogram = getCSM(XTempogram, XTempogram)
    if do_crema:
        XChordPitch = librosa.feature.stack_memory(
            chord_pitch, n_steps=wins_per_block, mode='edge').T
        DChordPitch = getCSMCosine(XChordPitch, XChordPitch)

    # Step 5: Run similarity network fusion
    FeatureNames = ['MFCCs', 'Chromas', 'Tempogram']
    Ds = [DMFCC, DChroma, DTempogram]
    if do_crema:
        FeatureNames.append('Crema')
        Ds.append(DChordPitch)
    # Edge case: If it's too small, zeropad SSMs
    for i, Di in enumerate(Ds):
        if Di.shape[0] < 2*K:
            D = np.zeros((2*K, 2*K))
            D[0:Di.shape[0], 0:Di.shape[1]] = Di
            Ds[i] = D
    pK = K
    if K == -1:
        pK = int(np.round(2*np.log(Ds[0].shape[0])/np.log(2)))
        print("Autotuned K = %i" % pK)
    # Do fusion on all features
    Ws = [getW(D, pK) for D in Ds]
    if REC_SMOOTH > 0:
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Ws = [df(W, size=(1, REC_SMOOTH)) for W in Ws]

    WFused = doSimilarityFusionWs(Ws, K=pK, niters=niters,
                                  reg_diag=reg_diag, reg_neighbs=reg_neighbs,
                                  do_animation=do_animation, PlotNames=FeatureNames,
                                  PlotExtents=[times[0], times[-1]])
    WsDict = {}
    for n, W in zip(FeatureNames, Ws):
        WsDict[n] = W
    WsDict['Fused'] = WFused
    # Do fusion with only Chroma and MFCC
    WsDict['Fused MFCC/Chroma'] = doSimilarityFusionWs(Ws[0:2], K=pK, niters=niters,
                                                       reg_diag=reg_diag, reg_neighbs=reg_neighbs)
    if do_crema:
        # Do fusion with tempograms and Crema if Crema is available
        WsDict['Fused Tgram/Crema'] = doSimilarityFusionWs(Ws[2::], K=pK, niters=niters,
                                                           reg_diag=reg_diag, reg_neighbs=reg_neighbs)
        # Do fusion with MFCC and Crema
        WsDict['Fused MFCC/Crema'] = doSimilarityFusionWs([Ws[0], Ws[-1]], K=pK, niters=niters,
                                                          reg_diag=reg_diag, reg_neighbs=reg_neighbs)
        # Do fusion with MFCC, Chroma, and Crema
        WsDict['Fused MFCC/Chroma/Crema'] = doSimilarityFusionWs([Ws[0], Ws[1], Ws[-1]], K=pK, niters=niters,
                                                                 reg_diag=reg_diag, reg_neighbs=reg_neighbs)

    return {'Ws': WsDict, 'times': times, 'K': pK}


def ssmFromOpt(opt, saveFile=None):
    res = getFusedSimilarity(opt.filename, sr=opt.sr,
                             hop_length=opt.hop_length, win_fac=opt.win_fac, wins_per_block=opt.wins_per_block,
                             K=opt.K, reg_diag=opt.reg_diag, reg_neighbs=opt.reg_neighbs, niters=opt.niters,
                             do_animation=False, plot_result=False, do_crema=False)
    if saveFile is not None:
        sio.savemat(saveFile, res)
    return res


class Opt:

    def __init__(self, filename, sr=22050, hop_length=512, win_fac=10, wins_per_block=20, K=5, reg_diag=1.0, reg_neighbs=0.5, niters=3):
        self.filename = filename
        self.sr = sr
        self.hop_length = hop_length
        self.win_fac = win_fac
        self.wins_per_block = wins_per_block
        self.K = K
        self.reg_diag = reg_diag
        self.reg_neighbs = reg_neighbs
        self.niters = niters

# opt = Opt('FDU/MIR/dataset/melodyExtraction/RWC/RWC-MDB-P-2001/RWC研究用音楽データベース/01 永遠のレプリカ.wav')
# res = ssmFromOpt(opt)
# times = res['times']
# fusedSSM = res['Ws']['Fused']
