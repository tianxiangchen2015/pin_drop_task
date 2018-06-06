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

from utils import *
from train_utils import *
from evaluation_measures import get_f_measure_by_class, select_threshold
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

X_train_fn, y_train = load_train_weak()
X_test_fn, y_test = load_validation_data()
y_train_one_hot = mlb.fit_transform(y_train)
y_test_one_hot = mlb.transform(y_test)
batch_size = 32
mode=3
# Create audio generator
audio_gen = DataGenerator(batch_size=batch_size, data_list=(X_train_fn, y_train_one_hot), mode=mode)
test_gen = AudioGenerator(batch_size=batch_size, fns=X_test_fn, labels=y_test_one_hot, mode=mode)

classes_num = 10
dropout_rate = 0.3

l, Sxx = audio_gen.rnd_one_sample()
test_step = test_gen.get_train_test_num() // batch_size
num_train, num_valid = audio_gen.get_train_test_num()
step_per_epoch = num_train // batch_size
validation_step = num_valid // batch_size
image_shape = Sxx.shape
# Attention CNN
model, model_name = base_model_5(image_shape, classes_num, dropout_rate)
print (model.summary())


# opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=1e-4)
#model.compile(optimizer='Adam', loss=[losses.binary_crossentropy, losses.binary_crossentropy],
#              loss_weights=[0.5, 0.5])
model.compile(optimizer='Adam', loss=losses.binary_crossentropy)
model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=step_per_epoch,
                          epochs=150, validation_data=audio_gen.next_test(), validation_steps=validation_step,
                          verbose=1)
print ("mode %d" % mode)
model.save('models/{0}_mode_{1}.h5'.format(model_name, str(mode)))

X_valid, y_valid = audio_gen.get_test()
y_valid = np.array(y_valid)

y_pred = model.predict(X_valid)
best_threshold = select_threshold(y_pred, y_valid)
valid_f_1 = get_f_measure_by_class(model, 10, audio_gen.next_test(), validation_step, thresholds=best_threshold)
test_f_1 = get_f_measure_by_class(model, 10, test_gen.next_train(), test_step, thresholds=best_threshold)

with open('results/f1_score_2018_weak.txt', 'ab') as fw:
    fw.write('model name: %s mode: %d valid_f1: %f test_f1: %f (optimized)\n' % (model_name, mode, valid_f_1, test_f_1))


