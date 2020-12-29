
# -*- coding: utf-8 -*-
import os
import sys
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import librosa


from madmom.audio.signal import *
from pysndfx import AudioEffectsChain

def randAudioAugment():
    fx = (AudioEffectsChain())
    effect = [random.randint(0, 1) for i in range(6)]
    
    if effect[0] == 1:  # lowshelf
        randGain = random.randint(0, 12) * random.choice([-1, 1])
        randFreq = random.randint(20, 300)
        randSlop = random.uniform(1, 7)/10  # 0.1~0.7
        fx.lowshelf(gain=randGain, frequency=randFreq, slope=randSlop)
    if effect[1] == 1:  # highshelf
        randGain = random.randint(0, 12) * random.choice([-1, 1])
        randFreq = random.randint(1000, 3000)
        randSlop = random.uniform(1, 7)/10  # 0.1~0.7
        fx.highshelf(gain=randGain, frequency=randFreq, slope=randSlop)
    if effect[2] == 1:  # equalizer
        randFreq = random.randint(100, 3000)
        randQ = random.uniform(5, 15)/10  # 0.5~1.5
        randDB = random.randint(0, 6) * random.choice([-1, 1])
        fx.equalizer(frequency=randFreq, q=randQ, db=randDB)
    if effect[3] == 1:  # overdrive
        randGain = random.randint(3, 7)
        fx.overdrive(gain=randGain, colour=40)
    if effect[4] == 1:  # phaser
        fx.phaser(gain_in=0.9, gain_out=0.8, delay=1,
                  decay=0.25, speed=2, triangular=False)
    if effect[5] == 1:  # reverb
        randReverb = random.randint(30, 70)
        randDamp = random.randint(30, 70)
        randRoom = random.randint(30, 70)
        randWet = random.randint(1, 6)
        fx.reverb(reverberance=randReverb, hf_damping=randDamp, room_scale=randRoom,
                  stereo_depth=100, pre_delay=20, wet_gain=randWet, wet_only=False)    
    return fx


def save_STFT(file_name, file_path):
    # y, sr = librosa.load(file_name, sr=8000)
    y = Signal(file_path, sample_rate=8000, dtype=np.float32, num_channels=1)
    S = librosa.core.stft(y, n_fft=1024, hop_length=80*1, win_length=1024)
    x_spec = np.abs(S)
    x_spec = librosa.core.power_to_db(x_spec, ref=np.max)
    x_spec = x_spec.astype(np.float32)
    PATH_spec = '/data/FMA_l/original/features/'
    if not os.path.exists(os.path.dirname(PATH_spec)):
        os.makedirs(os.path.dirname(PATH_spec))
    np.save(PATH_spec + file_name, x_spec)


def save_STFT_RAA(file_name, file_path):
    fx = randAudioAugment()
    y = Signal(file_path, sample_rate=8000, dtype=np.float32, num_channels=1)
    y_raa = fx(y)
    S_raa = librosa.core.stft(
        y_raa, n_fft=1024, hop_length=80*1, win_length=1024)
    x_spec_r = np.abs(S_raa)
    x_spec_r = librosa.core.power_to_db(x_spec_r, ref=np.max)
    x_spec_r = x_spec_r.astype(np.float32)
    
    PATH_spec_r = '/data/FMA_l/randAugment/features/'
    if not os.path.exists(os.path.dirname(PATH_spec_r)):
        os.makedirs(os.path.dirname(PATH_spec_r))
    np.save(PATH_spec_r + file_name, x_spec_r)

