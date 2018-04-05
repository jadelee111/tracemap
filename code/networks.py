#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:34:57 2018

@author: chen
"""

import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, Dense, Deconvolution3D,Flatten
from keras.layers import concatenate
from keras.optimizers import Adam
from metrics import jaccard_coef,jaccard_coef_loss,binary_crossentropy_weighted,jaccard_coef_int

K.set_image_data_format("channels_first")



def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='valid', strides=(1, 1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides,kernel_initializer = 'he_normal')(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)
    
def UndirectedSimpleNetwork(input_shape, pool_size=(2, 2, 2), n_labels=48, initial_learning_rate=0.00001, depth = 2,
                  n_base_filters=8, batch_normalization=False):
    '''the network takes in a volume and predicts the direction for the next trace point
    '''
    inputs = Input(input_shape)
    current_layer = inputs
    layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters,
                                          batch_normalization=batch_normalization)
    layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters,
                                          batch_normalization=batch_normalization)
    layer3 = MaxPooling3D(pool_size=pool_size)(layer2)
    
    layer4 = create_convolution_block(input_layer=layer3, n_filters=n_base_filters*2,
                                          batch_normalization=batch_normalization)
    layer5 = create_convolution_block(input_layer=layer4, n_filters=n_base_filters*2,
                                          batch_normalization=batch_normalization)
    layer6 = MaxPooling3D(pool_size=pool_size)(layer5)
    flat_layer = Flatten()(layer6)
    dense_layer1 = Dense(400, activation='relu', name='fc1')(flat_layer)
    dense_layer2 = Dense(n_labels,activation = 'sigmoid',name='fc2')(dense_layer1)
    model = Model(inputs=inputs, outputs=dense_layer2)
    metrics=['acc',jaccard_coef_int]
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=binary_crossentropy_weighted,metrics=metrics)
    print(model.summary())
    return model
    
def DirectedSimpleNetwork(input_shape, pool_size=(2, 2, 2), n_labels=48, initial_learning_rate=0.00001, depth = 2,
                  n_base_filters=8, batch_normalization=False):
    '''the network takes in a volume and the unit direction from parent node 
        predicts the direction for the next trace point
    '''
    input1 = Input(input_shape)
    current_layer = input1
    layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters,
                                          batch_normalization=batch_normalization)
    layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters,
                                          batch_normalization=batch_normalization)
    layer3 = MaxPooling3D(pool_size=pool_size)(layer2)
    
    layer4 = create_convolution_block(input_layer=layer3, n_filters=n_base_filters*2,
                                          batch_normalization=batch_normalization)
    layer5 = create_convolution_block(input_layer=layer4, n_filters=n_base_filters*2,
                                          batch_normalization=batch_normalization)
    layer6 = MaxPooling3D(pool_size=pool_size)(layer5)
    flat_layer = Flatten()(layer6)
    input2 = Input((48,))
    concat = concatenate([input2,flat_layer],axis=1)
    dense_layer1 = Dense(400, activation='relu', name='fc1')(concat)
    dense_layer2 = Dense(n_labels,activation = 'sigmoid',name='prediction')(dense_layer1)
    model = Model(inputs=[input1,input2], outputs=dense_layer2)
    
    metrics=['acc',jaccard_coef_int]
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=jaccard_coef_loss,metrics=metrics)
    print(model.summary())
    return model