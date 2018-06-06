import numpy as np
import scipy.io.wavfile as wavfiles
from gammatone.gtgram import gtgram
from utils import load_train_weak
from multiprocessing import Pool
from functools import partial
import pandas as pd
from utils import load_validation_data


def gen_gamatone(filename, feat_path):

    fs, wav = wavfiles.read(filename)
    fn = filename.split('/')[-1]
    print(fn)
    if len(wav.shape) > 1:
        wav = wav[:, 0]
    if wav.shape[0] < 441000:
        pad_with = 441000 - wav.shape[0]
        wav = np.pad(wav, (0, pad_with), 'constant', constant_values=(0))
    elif wav.shape[0] > 441000:
        wav = wav[0:441000]
    gtg = gtgram(wav, fs=fs, window_time=0.04, hop_time=0.02, channels=40, f_min=50).T

    np.save(feat_path + fn, gtg)

    return


if __name__ == '__main__':
    feature_path = 'audio/gammatone_feat/train/'
    '''
    path = 'audio/test/'
    data_list = 'metadata/test/test.csv'
    df = pd.read_csv(data_list, sep='\t')
    X_train_fn = list(df.filename.values)
    '''
    X_train_fn, _ = load_validation_data()
    pool = Pool(16)
    pool.map(partial(gen_gamatone, feat_path=feature_path), X_train_fn)
    pool.close()

