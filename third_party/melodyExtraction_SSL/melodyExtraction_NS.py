# -*- coding: utf-8 -*-

import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
import glob

from model import *
from featureExtraction import *


def melodyExtraction_NS(file_name, output_path, gpu_index):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpu_index is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    note_res = 8
    pitch_range = np.arange(40, 95 + 1.0 / note_res, 1.0 / note_res)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    """  Features extraction"""
    X_test, X_spec = spec_extraction(file_name=file_name, win_size=31)

    """  melody predict"""
    model = melody_ResNet()
    model.load_weights("./weights/ResNet_NS.hdf5")
    y_predict = model.predict(X_test, batch_size=64, verbose=1)

    y_shape = y_predict.shape
    num_total_frame = y_shape[0] * y_shape[1]
    est_pitch = np.zeros(num_total_frame)
    index_predict = np.zeros(num_total_frame)

    y_predict = np.reshape(y_predict, (num_total_frame, y_shape[2]))

    for i in range(num_total_frame):
        index_predict[i] = np.argmax(y_predict[i, :])
        pitch_MIDI = pitch_range[np.int32(index_predict[i])]
        if pitch_MIDI >= 45 and pitch_MIDI <= 95:
            est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.0) * 440

    """ save results """
    PATH_est_pitch = output_path + "pitch_" + file_name.split("/")[-1] + ".txt"
    if not os.path.exists(os.path.dirname(PATH_est_pitch)):
        os.makedirs(os.path.dirname(PATH_est_pitch))
    f = open(PATH_est_pitch, "w")
    for j in range(len(est_pitch)):
        est = "%.2f %.4f\n" % (0.01 * j, est_pitch[j])
        f.write(est)
    f.close()

    est_arr = np.loadtxt(PATH_est_pitch)
    return est_arr


def parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-p",
        "--filepath",
        help="Path to input audio (default: %(default)s",
        type=str,
        default="test_audio_file.mp4",
    )
    p.add_argument(
        "-o",
        "--output_dir",
        help="Path to output folder (default: %(default)s",
        type=str,
        default="./results/",
    )
    p.add_argument(
        "-gpu",
        "--gpu_index",
        help="Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s",
        type=int,
        default=None,
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parser()
    melodyExtraction_NS(
        file_name=args.filepath, output_path=args.output_dir, gpu_index=args.gpu_index
    )
