import numpy as np
import random
import librosa
# import webrtcvad
import scipy.io.wavfile as wavfiles
from scipy.signal import spectrogram
from python_speech_features import mfcc, logfbank
#from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
#from tensorflow.python.ops import io_ops


class DataGenerator():
    def __init__(self, batch_size=32, vad_mode=None, data_list=None):
        """
        Params:
            step (int): Step size in milliseconds between windows (for spectrogram ONLY)
            window (int): FFT window size in milliseconds (for spectrogram ONLY)
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned (for spectrogram ONLY)
        """

        self.train_index = 0
        self.valid_index = 0
        self.test_index = 0
        self.vad_mode = vad_mode
        self.batch_size = batch_size
        self.data_list = data_list
        self.num_classes = 10
        self.train_labels, self.train_fns, self.test_labels, self.test_fns = self.train_valid_split()

    def gen_spectrogram(self, filenames):
        """ For a given audio clip, calculate the corresponding feature
        Params:
            audio_clip (str): Path to the audio clip
        """
        x_data = []
        for filename in filenames:
            fs, wav = wavfiles.read(filename)
            # print(wav.shape, filename)
            if len(wav.shape) > 1:
                wav = wav[:,0]
            if wav.shape[0] < 441000:
                pad_with = 441000 - wav.shape[0]
                wav = np.pad(wav, (0, pad_with), 'constant', constant_values=(0))
            elif wav.shape[0] > 441000:
                wav = wav[0:441000]
            Sxx = logfbank(wav, fs, winlen=0.04, winstep=0.02, nfft=2048, nfilt=40)
            x_data.append(Sxx.reshape(1, Sxx.shape[0], Sxx.shape[1], 1))
            
        return np.vstack(x_data)
        
    
    def train_valid_split(self):
        
        fns, labels = self.data_list
        s_labels, s_fns = shuffle_data(labels, fns, rnd_seed=0)
        train_size = int(len(s_labels) * 0.7)
#         self.train_labels = s_labels[0:train_size]
#         self.train_fns = s_fns[0:train_size]
#         self.train_durations = s_durations[0:train_size]
#         self.test_labels = s_labels[train_size::]
#         self.test_fns = s_fns[train_size::]
#         self.test_durations = s_durations[train_size::]
        
        return s_labels[0:train_size], s_fns[0:train_size], s_labels[train_size::], s_fns[train_size::]
    
    def shuffle_data_by_partition(self, partition):
        if partition == 'train':
            self.train_labels, self.train_fns = shuffle_data(self.train_labels, self.train_fns)
        elif partition == 'test':
            self.test_labels, self.test_fns = shuffle_data(self.test_labels, self.test_fns)
    
    def get_next(self, partition):
        if partition=='train':
            cur_index = self.train_index
            audio_files = self.train_fns
            labels = self.train_labels
        elif partition=='test':
            cur_index = self.test_index
            audio_files = self.test_fns
            labels = self.test_labels
        X_labels = labels[cur_index: cur_index+self.batch_size]
        filenames = audio_files[cur_index: cur_index+self.batch_size]
        X_data = self.gen_spectrogram(filenames)

        outputs = np.vstack(X_labels)
        # inputs = np.vstack(X_data)
        inputs = X_data

        return (inputs, outputs)
    
    def next_train(self):
        while True:
            ret = self.get_next('train')
            self.train_index += self.batch_size
            if self.train_index > len(self.train_labels) - self.batch_size:
                self.train_index = 0
                self.shuffle_data_by_partition('train')
            yield ret
    
    def next_test(self):
        while True:
            ret = self.get_next('test')
            self.test_index += self.batch_size
            if self.test_index > len(self.test_labels) - self.batch_size:
                self.test_index = 0
                self.shuffle_data_by_partition('test')
            yield ret
    
    def get_test(self):
        """
        get the entire test data for prediction
        """
        self.shuffle_data_by_partition('test')
        features = self.gen_spectrogram(self.test_fns)
        texts = np.argmax(self.test_labels, axis=1)

        return features, texts
   
    def rnd_one_sample(self):

        rnd = np.random.choice(len(self.test_labels), 1)[0]
        Sxx = self.gen_spectrogram([self.test_fns[rnd]])
        return self.test_labels, Sxx

    def get_train_test_num(self):
        return len(self.train_labels), len(self.test_labels)


def shuffle_data(labels, fns, rnd_seed=None):
    np.random.seed(rnd_seed)
    p = np.random.permutation(len(fns))
    fns_shuffle = [fns[i] for i in p] 
    labels_shuffle = [labels[i] for i in p]
    return labels_shuffle, fns_shuffle

