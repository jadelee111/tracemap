#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:10:32 2018

@author: chen
"""

import numpy as np
import data_utils as utils
from generator import data_generator_undirected,sample_nodes_truth
from networks import UndirectedSimpleNetwork, DirectedSimpleNetwork
from keras.callbacks import ModelCheckpoint

train_dir = '../data/train'
val_dir = '../data/val'
test_dir = '../data/test'
input_shape = ((1,24,24,24))

traindatalist = utils.image_ids_in(train_dir)
valdatalist = utils.image_ids_in(val_dir)
traindatalist2 = traindatalist[0]

train_generator = data_generator_undirected(train_dir,traindatalist, n_label= 48,batch_size= 10,num_nodes_per_img =50)
val_generator = data_generator_undirected(val_dir,valdatalist, n_label= 48,batch_size= 10,num_nodes_per_img =50)

model2 = DirectedSimpleNetwork(input_shape, pool_size=(2, 2, 2), n_labels=48, initial_learning_rate=0.00001, depth = 2,
                  n_base_filters=8, batch_normalization=False)

#checkpointer = ModelCheckpoint(filepath='../weights/tracemap.h5', verbose=1, save_best_only=True)
#model.load_weights('../weights/tracemap.h5')
#for i in range(5):
#    model.fit_generator(data_generator(train_dir,traindatalist), 
#                   steps_per_epoch=1000, epochs=1, shuffle=True, 
#                   validation_data=data_generator(val_dir,valdatalist), 
#                   validation_steps=200, verbose=1, callbacks=[checkpointer])
#    model.save_weights('../weights/tracemap'+str(i)+'.h5')
#    
#for x,y in data_generator(train_dir,traindatalist):
#    pass