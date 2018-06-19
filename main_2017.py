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

from utils import *
from train_utils import *
from evaluation_measures import *
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
import pandas as pd
import json


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


df = pd.read_csv('metadata/2017/train_subset.csv', sep=',', quotechar='"', dtype='str')
df_test = pd.read_csv('metadata/2017/test_set.csv', sep=',', quotechar='"', dtype='str')
df_eval = pd.read_csv('metadata/2017/evaluation_set.csv', sep=',', quotechar='"', dtype='str')
X_train_fn, y_train = df.filename.values, df.label.values
X_test_fn, y_test = df_test.filename.values, df_test.label.values
X_eval_fn, y_eval = df_eval.filename.values, df_eval.label.values

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
y_test_one_hot = mlb.transform(y_int_test)

y_int_eval = []
for item in y_eval:
    int_label = []
    labels = item.split(',')
    for l in labels:
        int_label.append(hash_dict[l])
    y_int_eval.append(int_label)
y_eval_one_hot = mlb.transform(y_int_eval)

classes_num = 17
dropout_rate = 0.5
mode=4
batch_size = 32
n_epoch = 200
interval = 10
feat_path = 'audio_2017/gammatone_feat/train/'
# Create audio generator
for i in range(1):

    audio_gen = AudioGenerator(batch_size=batch_size, feat_path=feat_path, fns=X_train_fn, labels=y_train_one_hot, mode=mode)
    valid_gen = AudioGenerator(batch_size=batch_size, feat_path=feat_path, fns=X_test_fn, labels=y_test_one_hot, mode=mode)
    eval_gen = AudioGenerator(batch_size=batch_size, feat_path=feat_path, fns=X_eval_fn, labels=y_eval_one_hot, mode=mode)
    l, Sxx = audio_gen.rnd_one_sample()

    num_train = audio_gen.get_train_test_num()
    num_test = valid_gen.get_train_test_num()
    print(num_train, num_test)
    step_per_epoch = num_train // batch_size
    validation_step = num_test // batch_size
    image_shape = Sxx.shape
    print(image_shape)

    model, model_name = base_model_4(image_shape, classes_num, dropout_rate)
    print (model.summary())
    class_weights = calculating_class_weights(y_train_one_hot)

    X_valid, y_valid = valid_gen.get_test()
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
                              initial_epoch=i, epochs=end_epoch, validation_data=valid_gen.next_train(), validation_steps=validation_step,
                              verbose=1)
        y_pred = model.predict(X_valid)
        valid_f_1, p_valid, r_valid, _, _, _, _ = get_metrics_by_class(y_pred, y_valid, 17, thresholds=0.5)
        print("valid_f1: %f, valid_precision: %f, valid_recall: %f" % (valid_f_1, p_valid, r_valid))
    #print ("mode %d" % mode)
    # model.save('models/{0}_mode_{1}_weighted.h5'.format(model_name, str(mode)))

    with open('results/base_model_4_2017/{0}_train_loss_mode_{1}_{2}.txt'.format(model_name, str(mode), str(i)), 'w') as outfile:
        json.dump(history.history, outfile)
    #model = load_model('models/{0}_mode_{1}_weighted.h5'.format(model_name, str(mode)))
    X_test, y_test_true = eval_gen.get_test()
    y_valid = np.array(y_valid)

    y_pred = model.predict(X_valid)
    np.save('results/base_model_4_2017/{0}_valid_predict_mode_{1}_{2}'.format(model_name, str(mode), str(i)), y_pred)
    np.save('results/base_model_4_2017/{0}_valid_true_mode_{1}_{2}'.format(model_name, str(mode), str(i)), y_valid)
    best_threshold = select_threshold(y_pred, y_valid)
    print (best_threshold)
    y_test_pred = model.predict(X_test)
    np.save('results/base_model_4_2017/{0}_test_predict_mode_{1}_{2}'.format(model_name, str(mode), str(i)), y_test_pred)
    np.save('results/base_model_4_2017/{0}_test_true_mode_{1}_{2}'.format(model_name, str(mode), str(i)), y_test_true)

    valid_f_1, p_valid, r_valid, tp_valid, fp_valid, fn_valid, tn_valid = get_metrics_by_class(y_pred, y_valid, classes_num, thresholds=best_threshold)
    test_f_1, p_test, r_test, tp_valid, fp_valid, fn_valid, tn_valid = get_metrics_by_class(y_test_pred, y_test_true, classes_num,  thresholds=best_threshold)

    with open('results/f1_score_2017.txt', 'ab') as fw:
        fw.write('model name: %s mode: %d valid_f1: %f valid_precision: %f valid_recall: %f ; test_f1: %f test_precision: %f test_recall: %f (optimized)\n' % (model_name, mode, valid_f_1, p_valid, r_valid, test_f_1, p_test, r_test))


