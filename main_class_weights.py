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
from evaluation_measures import *
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import json
import pandas as pd


def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss


mlb = MultiLabelBinarizer()
X_train_fn, y_train = load_train_weak()
X_test_fn, y_test = load_validation_data()
y_train_one_hot = mlb.fit_transform(y_train)
y_test_one_hot = mlb.transform(y_test)
batch_size = 32
mode=4
interval = 10
n_epoch = 150
# Create scaler
# Create audio generator

for i in range(1):
    audio_gen = DataGenerator(batch_size=batch_size, data_list=(X_train_fn, y_train_one_hot), mode=mode, seed=i)
    test_gen = AudioGenerator(batch_size=batch_size, fns=X_test_fn, labels=y_test_one_hot, mode=mode)
    
    classes_num = 10
    dropout_rate = 0.5
    
    l, Sxx = audio_gen.rnd_one_sample()
    test_step = test_gen.get_train_test_num() // batch_size
    num_train, num_valid = audio_gen.get_train_test_num()
    step_per_epoch = num_train // batch_size
    validation_step = num_valid // batch_size
    image_shape = Sxx.shape

    X_valid, y_valid = audio_gen.get_test()
    y_valid = np.array(y_valid)

    model, model_name = base_model_4(image_shape, classes_num, dropout_rate)
    print (model.summary())
    class_weights = calculating_class_weights(y_train_one_hot)
    
    # opt = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=1e-4)
    #model.compile(optimizer='Adam', loss=[losses.binary_crossentropy, losses.binary_crossentropy],
    #              loss_weights=[0.5, 0.5])
    model.compile(optimizer='Adam', loss=get_weighted_loss(class_weights))
    for i in range(0, n_epoch, interval):
        end_epoch = i + interval
        if end_epoch > n_epoch:
            end_epoch = n_epoch
        print "triaing epoch: %d" % i
        history = model.fit_generator(generator=audio_gen.next_train(), steps_per_epoch=step_per_epoch,
                              initial_epoch=i, epochs=end_epoch, validation_data=audio_gen.next_test(), validation_steps=validation_step,
                              verbose=1)
        y_pred = model.predict(X_valid)
        valid_f_1, p_valid, r_valid, _, _, _, _ = get_metrics_by_class(y_pred, y_valid, 10, thresholds=0.5)
        print("valid_f1: %f, valid_precision: %f, valid_recall: %f" % (valid_f_1, p_valid, r_valid)) 
    #model = load_model('models/{0}_mode_{1}_weighted.h5'.format(model_name, str(mode)))
    
    y_pred = model.predict(X_valid)
    np.save('results/base_model_4/bi_{0}_valid_predict_mode_{1}_{2}'.format(model_name, str(mode), str(i)), y_pred)
    np.save('results/base_model_4/bi_{0}_valid_true_mode_{1}_{2}'.format(model_name, str(mode), str(i)), y_valid)
    best_threshold = select_threshold(y_pred, y_valid)
    print best_threshold
    X_test, y_test_true = test_gen.get_test() 
    y_test_pred = model.predict(X_test)
    np.save('results/base_model_4/bi_{0}_test_predict_mode_{1}_{2}'.format(model_name, str(mode), str(i)), y_test_pred)
    np.save('results/base_model_4/bi_{0}_test_true_mode_{1}_{2}'.format(model_name, str(mode), str(i)), y_test_true)    

    valid_f_1, p_valid, r_valid, _, _, _, _ = get_metrics_by_class(y_pred, y_valid, 10, thresholds=best_threshold)
    test_f_1, p_test, r_test, _, _, _, _ = get_metrics_by_class(y_test_pred, y_test_true, 10,  thresholds=best_threshold)

    with open('results/base_model_4/bi_{0}_train_loss_mode_{1}_{2}.txt'.format(model_name, str(mode), str(i)), 'w') as outfile:
        json.dump(history.history, outfile)
 
    model.save('models/bi_{0}_mode_{1}.h5'.format(model_name, str(mode))) 

    with open('results/f1_score_2018.txt', 'ab') as fw:
        fw.write('model name: bi_%s mode: %d valid_f1: %f valid_precision: %f valid_recall: %f ; test_f1: %f test_precision: %f test_recall: %f (weighted)\n' % (model_name, mode, valid_f_1, p_valid, r_valid, test_f_1, p_test, r_test))


