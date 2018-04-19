import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import Model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate, Conv2D, MaxPooling2D)
from keras.initializers import lecun_normal
import numpy as np
from data_generator import DataGenerator
import pandas as pd
import tensorflow as tf
print (tf.__version__)


def load_train_weak():
    data_list = 'metadata/train/weak.csv'
    path = 'audio/train/weak/'
    df = pd.read_csv(data_list, sep='\t')
    df['filename'] = path + df['filename']
    fns = df.filename.values
    labels = df.event_labels.str.split(',').values

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    one_hot_labels = mlb.fit_transform(labels)

    return fns, one_hot_labels, one_hot_labels.shape[1]


X_train_fn, y_train, num_class = load_train_weak()
batch_size = 32
# Create audio generator
audio_gen = DataGenerator(data_list=(X_train_fn, y_train))

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
cla = Dense(classes_num, activation='sigmoid')(dense_a)
att = Dense(classes_num, activation='softmax')(dense_a)
output_layer = Lambda(attention_pooling, output_shape=pooling_shape)([cla, att])
model = Model(inputs=input_layer, outputs=output_layer)
print (model.summary())

opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=1e-4)
    #checkerpoint = ModelCheckpoint(model_path+'best_se_{}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.compile(optimizer=opt,
                  loss=losses.binary_crossentropy, metrics=[metrics.categorical_accuracy])
model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=step_per_epoch,
                          epochs=40, validation_data=audio_gen.next_test(), validation_steps=validation_step,
                          verbose=1)

# input_layer = Input(shape=(500, 40))
#
# # Decision level single attention
# hidden_units = 256
# a1 = Dense(hidden_units)(input_layer)
# a1 = BatchNormalization()(a1)
# a1 = Activation('relu')(a1)
# a1 = Dropout(drop_rate)(a1)
#
# a2 = Dense(hidden_units)(a1)
# a2 = BatchNormalization()(a2)
# a2 = Activation('relu')(a2)
# a2 = Dropout(drop_rate)(a2)
#
# a3 = Dense(hidden_units)(a2)
# a3 = BatchNormalization()(a3)
# a3 = Activation('relu')(a3)
# a3 = Dropout(drop_rate)(a3)
#
# # cla = Dense(classes_num, activation='sigmoid')(a3)
# # att = Dense(classes_num, activation='softmax')(a3)
# # output_layer = Lambda(
#     # attention_pooling, output_shape=pooling_shape)([cla, att])
# model = Model(inputs=input_layer, outputs=a3)
# print (model.summary())