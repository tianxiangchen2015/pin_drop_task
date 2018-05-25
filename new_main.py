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
from data_generator import DataGenerator
import pandas as pd
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from utils import load_train_weak
from train_utils import base_model_1, benchmark_model, base_model_2
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

X_train_fn, y_train = load_train_weak()
y_train_one_hot = mlb.fit_transform(y_train)
batch_size = 32
# Create audio generator
audio_gen = DataGenerator(data_list=(X_train_fn, y_train_one_hot), mode=2)


classes_num = 10
dropout_rate = 0.25

l, Sxx = audio_gen.rnd_one_sample()

num_train, num_test = audio_gen.get_train_test_num()
print(num_train, num_test)
step_per_epoch = num_train // batch_size
validation_step = num_test // batch_size
image_shape = Sxx.shape
print(image_shape)
# Attention CNN
# model = base_model_1(image_shape, classes_num, dropout_rate)
model = base_model_2(image_shape, classes_num, dropout_rate)
print (model.summary())


# opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=1e-4)
#model.compile(optimizer='Adam', loss=[losses.binary_crossentropy, losses.binary_crossentropy],
#              loss_weights=[0.5, 0.5])
model.compile(optimizer='Adam', loss=losses.binary_crossentropy)
model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=step_per_epoch,
                          epochs=60, validation_data=audio_gen.next_test(), validation_steps=validation_step,
                          verbose=1)
model.save('models/base_model_2.h5')



model = load_model('models/base_model_2_mode_2.h5')
# X_test, y_test = audio_gen.next_test()

from evaluation_measures import get_f_measure_by_class

macro_f_measure = get_f_measure_by_class(model, 10, audio_gen.next_test(), validation_step, thresholds=0.2)
print(macro_f_measure)

