from model import MusicHighlighter
from lib import *
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def extract(fs, length=30, save_score=True, save_thumbnail=True, save_wav=True):
    with tf.Session() as sess:
        model = MusicHighlighter()
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, "model/model")
        print(f"model restored")
        for f in fs:
            print(f"extracting:{f}")
            name = os.path.split(f)[-1][:-4]
            audio, spectrogram, duration = audio_read(f)
            n_chunk, remainder = np.divmod(duration, 3)
            chunk_spec = chunk(spectrogram, n_chunk)
            pos = positional_encoding(
                batch_size=1, n_pos=n_chunk, d_pos=model.dim_feature * 4
            )

            n_chunk = n_chunk.astype("int")
            chunk_spec = chunk_spec.astype("float")
            pos = pos.astype("float")

            attn_score = model.calculate(
                sess=sess, x=chunk_spec, pos_enc=pos, num_chunk=n_chunk
            )
            attn_score = np.repeat(attn_score, 3)
            attn_score = np.append(attn_score, np.zeros(remainder))

            # score
            attn_score = attn_score / attn_score.max()
            if save_score:
                np.save("{}_score.npy".format(name), attn_score)

            # thumbnail
            attn_score = attn_score.cumsum()
            attn_score = np.append(
                attn_score[length], attn_score[length:] - attn_score[:-length]
            )
            index = np.argmax(attn_score)
            highlight = [index, index + length]
            if save_thumbnail:
                print(f"{name}_highlight.npy")
                np.save(
                    "/home/beantowel/FDU/MIR/Projects/results_of_highlighter/{}_highlight.npy".format(
                        name
                    ),
                    highlight,
                )

            if save_wav:
                librosa.output.write_wav(
                    "{}_audio.wav".format(name),
                    audio[highlight[0] * 22050 : highlight[1] * 22050],
                    22050,
                )
            yield highlight


if __name__ == "__main__":
    fileDir = "/home/beantowel/FDU/MIR/dataset/CCM_Chorus/audio"
    files = os.listdir(fileDir)
    fs = [os.path.join(fileDir, f) for f in files]
    # fs = ['YOUR MP3 FILE NAME 1', 'YOUR MP3 FILE NAME 2']  # list
    highlights = list(
        extract(fs, length=30, save_score=False, save_thumbnail=True, save_wav=False)
    )
