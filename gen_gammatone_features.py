import numpy as np
import scipy.io.wavfile as wavfiles
from gammatone.gtgram import gtgram
from utils import load_train_weak
from multiprocessing import Pool
from functools import partial
import pandas as pd


def gen_gamatone(audio_path, feat_path, filename):

    fs, wav = wavfiles.read(audio_path + filename)
    print(filename)
    if len(wav.shape) > 1:
        wav = wav[:, 0]
    if wav.shape[0] < 441000:
        pad_with = 441000 - wav.shape[0]
        wav = np.pad(wav, (0, pad_with), 'constant', constant_values=(0))
    elif wav.shape[0] > 441000:
        wav = wav[0:441000]
    gtg = gtgram(wav, fs=fs, window_time=0.04, hop_time=0.02, channels=40, f_min=50).T

    np.save(feat_path + filename, gtg)

    return


if __name__ == '__main__':

    audio_path = 'audio/train/weak/'
    data_list = 'metadata/train/weak.csv'
    feature_path = 'audio/gammatone_feat/train/'
    df = pd.read_csv(data_list, sep='\t')
    X_train_fn = df.filename.values

    pool = Pool(16)
    pool.map(partial(gen_gamatone, audio_path=audio_path, feat_path=feature_path), X_train_fn)
    pool.close()

