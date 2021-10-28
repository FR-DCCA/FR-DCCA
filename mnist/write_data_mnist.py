#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function

import pickle as pickle

import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split

path = '../dataset/'

def read_noisy_mnist():
    data = scio.loadmat(path + 'MNIST')
    view1_train = data['X1']
    view2_train = data['X2']
    label_train = data['trainLabel']

    print(view1_train)
    print(view2_train)

    view1_train = np.asarray(view1_train, dtype=np.float32)
    view2_train = np.asarray(view2_train, dtype=np.float32)
    label_train = np.asarray(label_train, dtype=np.int32).reshape((len(label_train))) - 1

    view1_validation = data['XV1']
    view2_validation = data['XV2']
    label_valdation = data['tuneLabel']
    view1_validation = np.asarray(view1_validation, dtype=np.float32)
    view2_validation = np.asarray(view2_validation, dtype=np.float32)
    label_validation = np.asarray(label_valdation, dtype=np.int32).reshape((len(label_valdation))) - 1

    view1_test = data['XTe1']
    view2_test = data['XTe2']
    label_test = data['testLabel']
    view1_test = np.asarray(view1_test, dtype=np.float32)
    view2_test = np.asarray(view2_test, dtype=np.float32)
    label_test = np.asarray(label_test, dtype=np.int32).reshape((len(label_test))) - 1
    return view1_train, view2_train, label_train, \
    view1_validation, view2_validation, label_validation, \
    view1_test, view2_test, label_test



def write_data(data_set_name='mnist', seed=3):
    view1_train, view2_train, label_train, \
    view1_validation, view2_validation, label_validation, \
    view1_test, view2_test, label_test = read_noisy_mnist()

    print(label_test)

    # Dump Test ==========================================================================
    print("Test size = ", len(label_test))
    with open(path + data_set_name + '/test.pkl', 'wb') as f_test:
        pickle.dump((view1_test, view2_test, label_test), f_test, -1)

    # Dump Test ==========================================================================
    print("Validation size = ", len(label_validation))
    with open(path + data_set_name + '/validation.pkl', 'wb') as f_val:
        pickle.dump((view1_validation, view2_validation, label_validation), f_val, -1)

    # Dump Test ==========================================================================
    print("Train size = ", len(label_train))
    with open(path + data_set_name + '/train.pkl', 'wb') as f_train:
        pickle.dump((view1_train, view2_train, label_train), f_train, -1)
    print("Write Done!")


def test_write():
    name = 'mnist'
    write_data()
    print("\n\n\nRead data ===========================================================================")

    with open(path + name + '/' + 'test.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
    print("test size = ", len(train_labels), train_page_data.shape, train_link_data.shape)

    with open(path + name + '/' + '/train.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
    print("train size = ", len(train_labels), train_page_data.shape, train_link_data.shape)

    with open(path + name + '/' + '/validation.pkl', 'rb') as fp:
        train_page_data, train_link_data, train_labels = pickle.load(fp)
    print("validation size = ", len(train_labels), train_page_data.shape, train_link_data.shape)


if __name__ == "__main__":
    test_write()
