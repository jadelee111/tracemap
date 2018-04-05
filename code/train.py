#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:54:10 2018


@author: chen
"""
import numpy as np
import data_utils as utils
from generator import data_generator_undirected,data_generator_directed,sample_nodes_truth,data_generator_undirected_sampleweight
from networks import UndirectedSimpleNetwork, DirectedSimpleNetwork,UndirectedSimpleNetwork_sampleweight
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import jaccard_similarity_score

train_dir = '../data/train'
val_dir = '../data/val'
test_dir = '../data/test'
input_shape = ((1,24,24,24))

traindatalist = utils.image_ids_in(train_dir)
valdatalist = utils.image_ids_in(val_dir)
traindatalist2 = traindatalist[0]

#train_generator = data_generator_undirected(train_dir,traindatalist2, n_label= 48,batch_size= 10,num_nodes_per_img =50)
#val_generator = data_generator_undirected(val_dir,valdatalist, n_label= 48,batch_size= 10,num_nodes_per_img =50)

model = UndirectedSimpleNetwork(input_shape, pool_size=(2, 2, 2), n_labels=48, initial_learning_rate=0.00001, depth = 2,
                  n_base_filters=8, batch_normalization=False)

#model = UndirectedSimpleNetwork_sampleweight(input_shape, pool_size=(2, 2, 2), n_labels=48, initial_learning_rate=0.00001, depth = 2,
#                  n_base_filters=8, batch_normalization=False)

checkpointer = ModelCheckpoint(filepath='../weights/tracemap.h5', verbose=1, save_best_only=True)
model.load_weights('../weights/tracemap.h5')
for i in range(5):
    model.fit_generator(data_generator_undirected(train_dir,[traindatalist2],n_label= 48,batch_size= 10,num_nodes_per_img =50),
                   steps_per_epoch=200, epochs=1, shuffle=True, 
                   validation_data=data_generator_undirected(val_dir,valdatalist, n_label= 48,batch_size= 10,num_nodes_per_img =50), 
                   validation_steps=50, verbose=1, callbacks=[checkpointer])
#    #use sample weight
#    model.fit_generator(data_generator_undirected_sampleweight(train_dir,[traindatalist2],n_label= 48,batch_size= 10,num_nodes_per_img =50),
#                   steps_per_epoch=200, epochs=1, shuffle=True, 
#                   validation_data=data_generator_undirected_sampleweight(val_dir,valdatalist, n_label= 48,batch_size= 10,num_nodes_per_img =50), 
#                   validation_steps=50, verbose=1, callbacks=[checkpointer])    
#    
    model.save_weights('../weights/tracemap'+str(i)+'.h5')
    folderpath = train_dir
    data_file =traindatalist[0]
    img_filename = folderpath + '/'+ data_file +'/'+ data_file +'.tif'
    swc_filename = folderpath + '/'+ data_file +'/'+data_file +'.tif.v3dpbd.swc'
    num_sample_nodes =2
    imgs,labels,p_encoding,node_ids = sample_nodes_truth(swc_filename,img_filename,num_nodes_per_img=num_sample_nodes, 
                                     child_step = 1,vis_flag= False)
    n_ch,n_x,n_y,n_z = 1, 24,24,24
    batch=num_sample_nodes
    n_label=48
    x_patch = np.zeros((batch,n_ch,n_x,n_y,n_z))
    y_patch = np.zeros((batch,n_label))
    x_patch[:,0,2:-1,2:-1,2:-1] =np.array(imgs)
    y_patch = np.array(labels)
    vis_enlarge_ratio =5
    ypred = model.predict(x_patch,batch_size=10)
    ypred[ypred>0.5]=1
    ypred[ypred<0.5]=0
    scores=[]
    for j in range(len(labels)):
        score1= jaccard_similarity_score(labels[j],ypred[j,:])
        score2= utils.smoothed_jaccard(labels[j],ypred[j,:])
        scores.append((score1,score2))
    print(scores)
    print(np.where(ypred[0,:]>0))
    print(np.where(labels[0]))

#for x,y in data_generator_undirected(train_dir,traindatalist):
#    pass


