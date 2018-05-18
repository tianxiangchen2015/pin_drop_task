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


def base_model_1():

    input_layer = Input(shape=(image_shape[1], image_shape[2], 1))
    
    cnn = Conv2D(64, (3, 3), padding='valid')(input_layer)
    cnn = Activation('relu')(cnn)
    #cnn = MaxPooling2D((1, 4))(cnn)
    cnn = Conv2D(64, (3, 3), padding='valid')(cnn)
    cnn = Activation('relu')(cnn)
    #cnn = MaxPooling2D((1, 4))(cnn)
    cnn = Conv2D(64, (3, 3), padding='valid')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Conv2D(10, (493, 34), padding='valid')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = Reshape((1,10))(cnn)
    dense_a = Dense(128, activation='relu')(cnn)
    #dense_a = Lambda(global_average_pooling,output_shape=global_average_pooling_shape)(cnn)
    dense_b = Dense(128, activation='relu')(dense_a)
    cla = Dense(128, activation='linear')(dense_a)
    att = Dense(128, activation='sigmoid')(dense_a)
    dense_b = Lambda(attention_pooling, output_shape=pooling_shape)([cla, att])
    b1 = BatchNormalization()(dense_b)
    b1 = Activation(activation='relu')(b1)
    b1 = Dropout(dropout_rate)(b1)
    #b2 = Dense(512)(b1)
    #b2 = BatchNormalization()(b1)
    #b2 = Activation(activation='relu')(b2)
    #b2 = Dropout(dropout_rate)(b2)
    output_layer = Dense(classes_num, activation='sigmoid')(b1)
    model = Model(inputs=input_layer, outputs=output_layer)
    print (model.summary())

classes_num = 10
dropout_rate = 0.25

l, Sxx = audio_gen.rnd_one_sample()

num_train, num_test = audio_gen.get_train_test_num()
print(num_train, num_test)
step_per_epoch = num_train // batch_size
validation_step = num_test // batch_size
image_shape = Sxx.shape
print(image_shape)
'''
input_layer = Input(shape=(image_shape[1], image_shape[2], 1))
cnn = Conv2D(64, (3, 3), padding='same')(input_layer)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D((1, 5))(cnn)
cnn = Conv2D(64, (3, 3), padding='same')(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D((1, 4))(cnn)
cnn = Conv2D(64, (3, 3), padding='same')(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D((1, 2))(cnn)
cnn = Reshape((499,64))(cnn)

bi_gru = Bidirectional(GRU(128, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,return_sequences=True))(cnn)
bi_gru = Bidirectional(GRU(128, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,return_sequences=True))(bi_gru)
dense_a = TimeDistributed(Dense(10, activation='relu'))(bi_gru)
dense_a = Dropout(rate=dropout_rate)(dense_a)
dense_a = TimeDistributed(Dense(10, activation='sigmoid'))(dense_a)

# Weak output path
weak_dense_a = TimeDistributed(Dense(1, activation='sigmoid'))(dense_a)
flat = Flatten()(weak_dense_a)
weak_dense_b = Dense(32, activation='relu')(flat)
Dropout(rate=dropout_rate)
weak_out = Dense(classes_num, activation='sigmoid')(weak_dense_b)

# Strong output path
strong_out = TimeDistributed(Dense(classes_num, activation='sigmoid'))(dense_a)
model = Model(inputs=input_layer, outputs=weak_out)
# model = Model(inputs=input_layer, outputs=[weak_out, strong_out])
print (model.summary())
'''

# Attention CNN

input_layer = Input(shape=(image_shape[1], image_shape[2], 1))

cnn = Conv2D(64, (3, 3), padding='valid')(input_layer)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D((1, 2))(cnn)
cnn = Conv2D(64, (3, 3), padding='valid')(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D((1, 2))(cnn)
cnn = Conv2D(64, (3, 3), padding='valid')(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D((2, 1))(cnn)
#cnn = Conv2D(10, (493, 34), padding='valid')(cnn)
#cnn = Activation('relu')(cnn)
cnn = Reshape((246,34*64))(cnn)
bi_gru = LSTM(512, activation='tanh', recurrent_dropout=dropout_rate,return_sequences=True)(cnn)
dense_a = Dense(512, activation='relu')(bi_gru)
#dense_a = Lambda(global_average_pooling,output_shape=global_average_pooling_shape)(cnn)
dense_b = Dense(256, activation='relu')(dense_a)
cla = Dense(256, activation='linear')(dense_a)
att = Dense(256, activation='sigmoid')(dense_a)
dense_b = Lambda(attention_pooling, output_shape=pooling_shape)([cla, att])
b1 = BatchNormalization()(dense_b)
b1 = Activation(activation='relu')(b1)
b1 = Dropout(dropout_rate)(b1)
b2 = Dense(128)(b1)
b2 = BatchNormalization()(b1)
b2 = Activation(activation='relu')(b2)
b2 = Dropout(dropout_rate)(b2)
output_layer = Dense(classes_num, activation='sigmoid')(b2)
model = Model(inputs=input_layer, outputs=output_layer)
print (model.summary())


opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=1e-4)
#model.compile(optimizer='Adam', loss=[losses.binary_crossentropy, losses.binary_crossentropy],
#              loss_weights=[0.5, 0.5])
model.compile(optimizer='Adam', loss=losses.binary_crossentropy)
model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=step_per_epoch,
                          epochs=40, validation_data=audio_gen.next_test(), validation_steps=validation_step,
                          verbose=1)
model.save('models/event_detect_attention_3.h5')



model = load_model('models/event_detect_attention_3.h5')
# X_test, y_test = audio_gen.next_test()

from evaluation_measures import get_f_measure_by_class

macro_f_measure = get_f_measure_by_class(model, 10, audio_gen.next_test(), validation_step, thresholds=0.2)
print(macro_f_measure)

