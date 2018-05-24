import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import Model, load_model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate, Conv2D, MaxPooling2D, Reshape,
                          TimeDistributed, Flatten, Bidirectional, LSTM, GRU, merge, Permute)
from keras.initializers import lecun_normal
import numpy as np


INPUT_DIM = 2
TIME_STEPS = 499
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False


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


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def base_model_1(image_shape, classes_num, dropout_rate):
    input_layer = Input(shape=(image_shape[1], image_shape[2], 3))
    cnn = Conv2D(128, (3, 3), padding='valid')(input_layer)
    cnn = Activation('relu')(cnn)
    #cnn = MaxPooling2D((1, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='valid')(cnn)
    cnn = Activation('relu')(cnn)
    #cnn = MaxPooling2D((1, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='valid')(cnn)
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
    return model


def benchmark_model(image_shape, classes_num, dropout_rate):

    input_layer = Input(shape=(image_shape[1], image_shape[2], 3))
    cnn = Conv2D(128, (3, 3), padding='same')(input_layer)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 5))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 4))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 2))(cnn)
    cnn = Reshape((499,128))(cnn)
    bi_gru = Bidirectional(GRU(128, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,return_sequences=True))(cnn)
    bi_gru = Bidirectional(GRU(128, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,return_sequences=True))(bi_gru)
    dense_a = TimeDistributed(Dense(10, activation='relu'))(bi_gru)
    dense_a = Dropout(rate=0.3)(dense_a)
    dense_a = TimeDistributed(Dense(10, activation='sigmoid'))(dense_a)
    weak_dense_a = TimeDistributed(Dense(1, activation='sigmoid'))(dense_a)
    flat = Flatten()(weak_dense_a)
    weak_dense_b = Dense(32, activation='relu')(flat)
    Dropout(rate=0.3)
    weak_out = Dense(classes_num, activation='sigmoid')(weak_dense_b)
    model = Model(inputs=input_layer, outputs=weak_out)
    return model


def base_model_2(image_shape, classes_num, dropout_rate):

    input_layer = Input(shape=(image_shape[1], image_shape[2], 3))
    cnn = Conv2D(128, (3, 3), padding='same')(input_layer)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 2))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 2))(cnn)
    cnn = Conv2D(128, (3, 3), padding='same')(cnn)
    cnn = Activation('relu')(cnn)
    cnn = MaxPooling2D((1, 2))(cnn)
    
    #cnn = Conv2D(10, (493, 34), padding='valid')(cnn)
    #cnn = Activation('relu')(cnn)
    res_cnn = Reshape((499,5*128))(cnn)
    bi_gru = LSTM(512, recurrent_dropout=dropout_rate,return_sequences=True)(res_cnn)
    attention_mul = attention_3d_block(bi_gru)
    attention_mul = Flatten()(attention_mul)
    dense_a = Dense(512, activation='relu')(attention_mul)
    #dense_a = Lambda(global_average_pooling,output_shape=global_average_pooling_shape)(cnn)
    dense_b = Dense(256, activation='relu')(dense_a)
    b1 = BatchNormalization()(dense_b)
    b1 = Activation(activation='relu')(b1)
    b1 = Dropout(dropout_rate)(b1)
    b2 = Dense(128)(b1)
    b2 = BatchNormalization()(b1)
    b2 = Activation(activation='relu')(b2)
    b2 = Dropout(dropout_rate)(b2)
    output_layer = Dense(classes_num, activation='sigmoid')(b2)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model




