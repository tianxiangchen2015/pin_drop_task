import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import Model, load_model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate, Conv2D, MaxPooling2D)
from keras.initializers import lecun_normal
import numpy as np
from data_generator import DataGenerator
import pandas as pd
import tensorflow as tf
print (tf.__version__)

from utils import load_train_weak


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

X_train_fn, y_train = load_train_weak()
y_train_one_hot = mlb.fit_transform(y_train)
batch_size = 32
# Create audio generator
audio_gen = DataGenerator(data_list=(X_train_fn, y_train_one_hot))

def global_average_pooling(x):
    return K.mean(x, axis = (3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:3]

def average_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.mean(input, axis=1)


def max_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.max(input, axis=1)


def attention_pooling(inputs, **kwargs):
    [out, att] = inputs

    epsilon = 1e-7
    att = K.clip(att, epsilon, 1. - epsilon)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]

    return K.sum(out * normalized_att, axis=1)


def pooling_shape(input_shape):

    if isinstance(input_shape, list):
        (sample_num, time_steps, freq_bins) = input_shape[0]

    else:
        (sample_num, time_steps, freq_bins) = input_shape

    return (sample_num, freq_bins)


classes_num = 10
drop_rate = 0.2

l, Sxx = audio_gen.rnd_one_sample()

num_train, num_test = audio_gen.get_train_test_num()
print(num_train, num_test)
step_per_epoch = num_train // batch_size
validation_step = num_test // batch_size
image_shape = Sxx.shape
print(image_shape)

input_layer = Input(shape=(image_shape[1], image_shape[2], 1))

cnn = Conv2D(64, (3, 3), padding='same', kernel_initializer=lecun_normal(seed=None), activation='selu')
cnn = cnn(input_layer)
cnn = MaxPooling2D((1, 4))(cnn)
cnn = Conv2D(64, (3, 3), padding='same', kernel_initializer=lecun_normal(seed=None), activation='selu')(cnn)
cnn = MaxPooling2D((1, 4))(cnn)
cnn = Conv2D(64, (3, 3), padding='same', kernel_initializer=lecun_normal(seed=None), activation='selu')(cnn)
cnn = MaxPooling2D((1, 2))(cnn)
dense_a = Lambda(global_average_pooling,output_shape=global_average_pooling_shape)(cnn)
dense_b = Dense(128, kernel_initializer=lecun_normal(seed=None), activation='selu')(dense_a)
cla = Dense(classes_num, activation='sigmoid')(dense_b)
att = Dense(classes_num, activation='softmax')(dense_b)
output_layer = Lambda(attention_pooling, output_shape=pooling_shape)([cla, att])
model = Model(inputs=input_layer, outputs=output_layer)
print (model.summary())

opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=1e-4)
model.compile(optimizer=opt,
                  loss=losses.binary_crossentropy, metrics=[metrics.binary_crossentropy])
model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=step_per_epoch,
                          epochs=1, validation_data=audio_gen.next_test(), validation_steps=validation_step,
                          verbose=1)
model.save('models/event_detect_attention.h5')



model = load_model('models/event_detect_attention.h5')
# X_test, y_test = audio_gen.next_test()

from evaluation_measures import get_f_measure_by_class

macro_f_measure = get_f_measure_by_class(model, 10, audio_gen.next_test(), validation_step, thresholds=None)
print(macro_f_measure)
