import numpy as np 
import librosa
import scipy.io.wavfile as wavfiles
from scipy.signal import spectrogram
from python_speech_features import logfbank
from scipy import ndimage
import gammatone.gtgram



def shuffle_data(labels, fns, rnd_seed=None):
    np.random.seed(rnd_seed)
    p = np.random.permutation(len(fns))
    fns_shuffle = [fns[i] for i in p]
    labels_shuffle = [labels[i] for i in p]
    return labels_shuffle, fns_shuffle


class DataGenerator():
    def __init__(self, batch_size=32, vad_mode=None, data_list=None, mode=1):

        self.train_index = 0
        self.valid_index = 0
        self.test_index = 0
        self.vad_mode = vad_mode
        self.batch_size = batch_size
        self.data_list = data_list
        self.mode = mode
        self.num_classes = 10
        self.train_labels, self.train_fns, self.test_labels, self.test_fns = self.train_valid_split()


    def gen_feture(self, filenames):
        if self.mode == 1:
            x_data = self.gen_spectrogram(filenames)
        elif self.mode == 2:
            x_data = self.gen_delta_delta(filenames)
        elif self.mode == 3:
            x_data = self.gen_filtered_spec(filenames)
        elif self.mode == 4:
            x_data = self.gen_gamatone(filenames)


        return x_data

    def gen_spectrogram(self, filenames):
        x_data = []
        for filename in filenames:
            wav, fs = librosa.load(filename, sr=None)
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
     
    def gen_delta_delta(self, filenames):
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
            delta = librosa.feature.delta(Sxx, order=1)
            delta_2 = librosa.feature.delta(Sxx, order=2)
            data = np.dstack((Sxx, delta, delta_2))
            x_data.append(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))

        return np.vstack(x_data)

    def gen_filtered_spec(self, filenames):
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
            kernel = [-1, 2, -1]
            Sxx = logfbank(wav, fs, winlen=0.04, winstep=0.02, nfft=2048, nfilt=40)
            delta = ndimage.convolve1d(Sxx, weights=kernel, axis=1, mode='nearest')
            delta_2 = ndimage.convolve1d(Sxx, weights=kernel, axis=0, mode='nearest')
            data = np.dstack((Sxx, delta, delta_2))
            x_data.append(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))

        return np.vstack(x_data)

    def gen_gamatone(self, filenames):
        x_data = []
        for filename in filenames:
            fs, wav = wavfiles.read(filename)
            # print(wav.shape, filename)
            if len(wav.shape) > 1:
                wav = wav[:, 0]
            if wav.shape[0] < 441000:
                pad_with = 441000 - wav.shape[0]
                wav = np.pad(wav, (0, pad_with), 'constant', constant_values=(0))
            elif wav.shape[0] > 441000:
                wav = wav[0:441000]
            gtg = gammatone.gtgram(x=wav, fs=fs, window_time=0.04, hop_time=0.02, channels=40, f_min=50)
            delta = librosa.feature.delta(gtg, order=1)
            delta_2 = librosa.feature.delta(gtg, order=2)
            Sxx = logfbank(wav, fs, winlen=0.04, winstep=0.02, nfft=2048, nfilt=40)
            Sxx_delta = librosa.feature.delta(Sxx, order=1)
            Sxx_delta_2 = librosa.feature.delta(Sxx, order=2)
            data = np.dstack((gtg, delta, delta_2, Sxx, Sxx_delta, Sxx_delta_2))
            x_data.append(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))

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

        strong_labels = []

        for i in range(cur_index, cur_index + self.batch_size):
            l = np.array([labels[i], ] * 499)
            strong_labels.append(l)

        X_labels = labels[cur_index: cur_index+self.batch_size]
        filenames = audio_files[cur_index: cur_index+self.batch_size]
        if self.mode == 1:
            X_data = self.gen_spectrogram(filenames)
        elif self.mode == 2:
            X_data = self.gen_delta_delta(filenames)
        else:
            X_data = self.gen_filtered_spec(filenames)
        inputs = X_data
        outputs_weak = np.vstack(X_labels)
        outputs_strong = np.array(strong_labels)

        #return (inputs, [outputs_weak, outputs_strong])
        return inputs, outputs_weak
    
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
        self.shuffle_data_by_partition('test')
        features = self.gen_spectrogram(self.test_fns)
        texts = np.argmax(self.test_labels, axis=1)

        return features, texts
   
    def rnd_one_sample(self):

        rnd = np.random.choice(len(self.test_labels), 1)[0]
        if self.mode == 1:
            Sxx = self.gen_spectrogram([self.test_fns[rnd]])
        elif self.mode == 2:
            Sxx = self.gen_delta_delta([self.test_fns[rnd]])
        else:
            Sxx = self.gen_filtered_spec([self.test_fns[rnd]])
        return self.test_labels, Sxx

    def get_train_test_num(self):
        return len(self.train_labels), len(self.test_labels)


class AudioGenerator():
    def __init__(self, batch_size=32, labels=None, fns=None, mode=1):

        self.train_index = 0
        self.batch_size = batch_size
        self.train_labels = labels
        self.train_fns = fns
        self.mode = mode
        self.shuffle_data_by_partition()

    def gen_spectrogram(self, filenames):
        x_data = []
        for filename in filenames:
            fs, wav = wavfiles.read(filename)
            # print(wav.shape, filename)
            if len(wav.shape) > 1:
                wav = wav[:, 0]
            if wav.shape[0] < 441000:
                pad_with = 441000 - wav.shape[0]
                wav = np.pad(wav, (0, pad_with), 'constant', constant_values=(0))
            elif wav.shape[0] > 441000:
                wav = wav[0:441000]
            Sxx = logfbank(wav, fs, winlen=0.04, winstep=0.02, nfft=2048, nfilt=40)
            x_data.append(Sxx.reshape(1, Sxx.shape[0], Sxx.shape[1], 1))

        return np.vstack(x_data)

    def gen_delta_delta(self, filenames):
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
            delta = librosa.feature.delta(Sxx, order=1)
            delta_2 = librosa.feature.delta(Sxx, order=2)
            data = np.dstack((Sxx, delta, delta_2))
            x_data.append(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))

        return np.vstack(x_data)

    def gen_filtered_spec(self, filenames):
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
            kernel = [-1, 2, -1]
            Sxx = logfbank(wav, fs, winlen=0.04, winstep=0.02, nfft=2048, nfilt=40)
            delta = ndimage.convolve1d(Sxx, weights=kernel, axis=1, mode='nearest')
            delta_2 = ndimage.convolve1d(Sxx, weights=kernel, axis=0, mode='nearest')
            data = np.dstack((Sxx, delta, delta_2))
            x_data.append(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))

        return np.vstack(x_data)

    def shuffle_data_by_partition(self):
        self.train_labels, self.train_fns = shuffle_data(self.train_labels, self.train_fns)

    def get_next(self):
        cur_index = self.train_index
        audio_files = self.train_fns
        labels = self.train_labels
        X_labels = labels[cur_index: cur_index + self.batch_size]
        filenames = audio_files[cur_index: cur_index + self.batch_size]
        if self.mode == 1:
            X_data = self.gen_spectrogram(filenames)
        elif self.mode == 2:
            X_data = self.gen_delta_delta(filenames)
        else:
            X_data = self.gen_filtered_spec(filenames)
        outputs = np.vstack(X_labels)
        inputs = X_data
        self.train_index += self.batch_size
        #if self.train_index > len(self.train_labels) - self.batch_size:
        #    self.train_index = 0
        #    self.shuffle_data_by_partition()

        return inputs, outputs

    def next_train(self):
        while True:
            ret = self.get_next()
            self.train_index += self.batch_size
            if self.train_index > len(self.train_labels) - self.batch_size:
                self.train_index = 0
                self.shuffle_data_by_partition()
            yield ret

    def rnd_one_sample(self):

        rnd = np.random.choice(len(self.train_labels), 1)[0]
        if self.mode == 1:
            Sxx = self.gen_spectrogram([self.train_fns[rnd]])
        elif self.mode == 2:
            Sxx = self.gen_delta_delta([self.train_fns[rnd]])
        return self.train_labels, Sxx

    def get_train_test_num(self):
        return len(self.train_labels)


class SemiDataGenerator():
    def __init__(self, labeled_data, unlabeled_data, batch_size=32):
        self.batch_size = batch_size
        self.labeled_lbs, self.labeled_fns = labeled_data
        self.unlabeled_lbs, self.unlabeled_fns = unlabeled_data
        self.n_labels = len(self.labeled_fns)
        self.labeled_data_generator = AudioGenerator(batch_size=self.batch_size, labels=self.labeled_lbs, fns=self.labeled_fns)
        self.unlabeled_data_generator = AudioGenerator(batch_size=self.batch_size, labels=self.unlabeled_lbs, fns=self.unlabeled_fns)

    def next_batch(self):
        unlabeled_feats, _ = self.unlabeled_data_generator.get_next()
        if self.batch_size > self.n_labels:
            labeled_feats, labels = self.labeled_data_generator.get_next()
        else:
            labeled_feats, labels = self.labeled_data_generator.get_next()

        inputs = np.vstack([labeled_feats, unlabeled_feats])

        return inputs, labels
