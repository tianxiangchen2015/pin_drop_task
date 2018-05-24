import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import Model, load_model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate, Conv2D, MaxPooling2D, Reshape, 
                          TimeDistributed, Flatten, Bidirectional, LSTM, GRU)
from keras.initializers import lecun_normal
import numpy as np
from data_generator import DataGenerator, AudioGenerator
import pandas as pd
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
print K.floatx()
print K.epsilon()
print K.image_dim_ordering()
print K.image_data_format()
print K.backend()

from utils import load_train_weak
from train_utils import base_model_1, benchmark_model
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
import pandas as pd


classes_num = 17
dropout_rate = 0.25
batch_size = 64

df = pd.read_csv('metadata/2017/train_set.csv')
df_test = pd.read_csv('metadata/2017/test_set.csv')
X_train_fn, y_train = df.filename.values, df.label.values
X_test_fn, y_test = df_test.filename.values, df_test.label.values

hash_dict = {'/m/03qc9zr': 3, '/m/012ndj': 1, '/t/dd00134': 2, '/m/0199g': 0, '/m/02mfyn': 4, '/m/0k4j': 5, '/m/07r04': 6, '/m/04_sv': 7, '/m/01bjv': 8, '/m/0dgbq': 9, '/m/04qvtq': 10, '/m/07jdr': 11, '/m/02rhddq': 12, '/m/06_fw': 13, '/m/012n7d': 14, '/m/0284vy3': 15, '/m/05x_td': 16}

y_int_train = []
for item in y_train:
    int_label = []
    labels = item.split(',')
    for l in labels:
        int_label.append(hash_dict[l])
    y_int_train.append(int_label)
y_train_one_hot = mlb.fit_transform(y_int_train)

y_int_test = []
for item in y_test:
    int_label = []
    labels = item.split(',')
    for l in labels:
        int_label.append(hash_dict[l])
    y_int_test.append(int_label)
y_test_one_hot = mlb.fit_transform(y_int_test)


batch_size = 32
# Create audio generator
audio_gen = AudioGenerator(batch_size=batch_size, fns=X_train_fn, labels=y_train_one_hot)
valid_gen = AudioGenerator(batch_size=batch_size, fns=X_test_fn, labels=y_test_one_hot)
l, Sxx = audio_gen.rnd_one_sample()

num_train = audio_gen.get_train_test_num()
num_test = valid_gen.get_train_test_num()
print(num_train, num_test)
step_per_epoch = num_train // batch_size
validation_step = num_test // batch_size
image_shape = Sxx.shape
print(image_shape)


# Attention CNN
model = base_model_1(image_shape, classes_num, dropout_rate)
print (model.summary())


opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=1e-4)
#model.compile(optimizer='Adam', loss=[losses.binary_crossentropy, losses.binary_crossentropy],
#              loss_weights=[0.5, 0.5])
model.compile(optimizer='Adam', loss=losses.binary_crossentropy)
model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=step_per_epoch,
                          epochs=60, validation_data=valid_gen.next_train(), validation_steps=validation_step)
model.save('models/event_detect_attention_4.h5')



model = load_model('models/event_detect_attention_4.h5')
# X_test, y_test = audio_gen.next_test()
print "finish loading model"
from evaluation_measures import get_f_measure_by_class

macro_f_measure = get_f_measure_by_class(model, classes_num, valid_gen.next_train(), validation_step, thresholds=0.5)
print(macro_f_measure)

