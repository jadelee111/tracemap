#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:55:03 2018

@author: chen
"""
import numpy as np
import data_utils as utils
from generator import data_generator_undirected,data_generator_directed
from networks import UndirectedSimpleNetwork, DirectedSimpleNetwork
from keras.callbacks import ModelCheckpoint

train_dir = '../data/train'
val_dir = '../data/val'
test_dir = '../data/test'
input_shape = ((1,24,24,24))

traindatalist = utils.image_ids_in(train_dir)
valdatalist = utils.image_ids_in(val_dir)
traindatalist2 = traindatalist[0]
model_path = '../weights/tracemap.h5'

model = UndirectedSimpleNetwork(input_shape, pool_size=(2, 2, 2), n_labels=48, initial_learning_rate=0.00001, depth = 2,
                  n_base_filters=8, batch_normalization=False)

model.load_weights(model_path)

