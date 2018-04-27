import pandas as pd
import numpy as np


def load_train_weak():
    data_list = 'metadata/train/weak.csv'
    path = 'audio/train/weak/'
    df = pd.read_csv(data_list, sep='\t')
    df['filename'] = path + df['filename']
    fns = df.filename.values
    labels = df.event_labels.str.split(',').values

    return fns, labels


def load_unlabel_data():
    data_list = 'metadata/train/unlabel_in_domain.csv'
    missing_list = 'metadata/missing_files_unlabel_in_domain.csv'
    path = 'audio/train/unlabel_in_domain/'
    df = pd.read_csv(data_list, sep='\t')
    df_miss = pd.read_csv(missing_list, sep=',')
    fns_miss = df_miss.filename.values
    df = df[~df['filename'].isin(fns_miss)]
    df['filename'] = path + df['filename']
    fns = df.filename.values
    labels = np.zeros(fns.shape)

    return fns, labels


def load_validation_data():
    data_list = 'metadata/test/test.csv'
    missing_list = 'metadata/missing_files_test.csv'
    path = 'audio/test/'
    df = pd.read_csv(data_list, sep='\t')
    df_miss = pd.read_csv(missing_list, sep=',')
    fns_miss = df_miss.filename.values
    df = df[~df['filename'].isin(fns_miss)]
    df['filename'] = path + df['filename']
    fns = df.filename.values
    labels = df.event_label.str.split(',').values

    return fns, labels