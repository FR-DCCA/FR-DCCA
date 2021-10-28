#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import torch.utils.data as data
import pickle
import numpy as np

path = '../dataset/'



class MNISTValidation(data.Dataset):
    def __init__(self, set_name='test'):
        self.set_name = set_name
        self.processed_folder = path + 'mnist/'
        self.train_file = 'train.pkl'
        self.validation_file = 'validation.pkl'
        self.test_file = 'test.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)
        elif self.set_name == 'val':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)
        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.train_page_data, self.train_link_data, self.labels = pickle.load(fp)


        length = self.__len__()
        # print(self.set_name, "Data Len = ", length)

    def __getitem__(self, index):
        page, link, target = self.train_page_data[index], self.train_link_data[index], self.labels[index]
        return page, link, target

    def __len__(self):
        return len(self.train_link_data)
