#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:55:03 2018

@author: chen
"""
import numpy as np
import data_utils as utils
from generator import data_generator_undirected,data_generator_directed,sample_nodes_truth
from networks import UndirectedSimpleNetwork, DirectedSimpleNetwork
from keras.callbacks import ModelCheckpoint
from neuronTree import Treenode, Neurontree
from sklearn.metrics import jaccard_similarity_score

train_dir = '../data/train'
val_dir = '../data/val'
test_dir = '../data/test'
input_shape = ((1,24,24,24))

traindatalist = utils.image_ids_in(train_dir)
valdatalist = utils.image_ids_in(val_dir)
testdatalist = utils.image_ids_in(test_dir)
model_path = '../weights/tracemap.h5'

model1 = UndirectedSimpleNetwork(input_shape, pool_size=(2, 2, 2), n_labels=48, initial_learning_rate=0.00001, depth = 2,
                  n_base_filters=8, batch_normalization=False)

#model2 = DirectedSimpleNetwork(input_shape, pool_size=(2, 2, 2), n_labels=48, initial_learning_rate=0.00001, depth = 2,
#                  n_base_filters=8, batch_normalization=False)

#model1.load_weights(model_path)

folderpath = train_dir
data_file =traindatalist[0]
img_filename = folderpath + '/'+ data_file +'/'+ data_file +'.tif'
swc_filename = folderpath + '/'+ data_file +'/'+data_file +'.tif.v3dpbd.swc'
num_sample_nodes =5
imgs,labels,p_encoding,node_ids = sample_nodes_truth(swc_filename,img_filename,num_nodes_per_img=num_sample_nodes, 
                                 child_step = 1,vis_flag= False)
n_ch,n_x,n_y,n_z = 1, 24,24,24
batch=num_sample_nodes
n_label=48
x_patch = np.zeros((batch,n_ch,n_x,n_y,n_z))
y_patch = np.zeros((batch,n_label))
x2_patch = np.zeros((batch,n_label))
x_patch[:,0,2:-1,2:-1,2:-1] =np.array(imgs)
y_patch[:,:] = np.array(labels)
x2_patch[:,:] = np.array(p_encoding)
vis_enlarge_ratio =5
ypred = model1.predict(x_patch,batch_size=10)
ypred[ypred>0.5]=1
ypred[ypred<0.5]=0
scores=[]

for i in range(batch):
    score1= jaccard_similarity_score(labels[i],ypred[i,:])
    score2= utils.accuracy(labels[i],ypred[i,:])
    scores.append((score1,score2))
print(scores)
'''
sphere_vectors = utils.load_sphere_48_units()
m = np.where(ypred>0.5)
print('node_ids:',node_ids)
print(len(m[0]))
#visualize ypred in green
for i,node_id in enumerate(node_ids):
    fname = '../results/trace%d_1.swc'%(node_id)
    nt = utils.read_swc_get_nt(fname)
    center_node  = nt.GetNode(1)
    out_swc = []
    out_swc.append(center_node)
    max_idx =np.where(ypred[i,:]>0.5)
    if len(max_idx[0])>0:
        #the direction of unit vector
        for j in range(len(max_idx)):
            unit_vector =sphere_vectors[max_idx[j],:]
            enlarged_vector = unit_vector[j]*vis_enlarge_ratio + [center_node.x,center_node.y,center_node.z]
            node = Treenode(j+2,5,enlarged_vector[0],enlarged_vector[1],enlarged_vector[2],0,1)
            out_swc.append(node)    
        fn_traceswc = '../results/pred%d.swc'%(node_id)
        utils.write_swc(fn_traceswc,out_swc,5)
        fn_anno = '../results/%d_1.ano'%(node_id)
        file = open(fn_anno,"a")
        file.write("SWCFILE=%s\n"%fn_traceswc)
        file.close()
    else:
        print("i: %d, node_id: %d All zeros\n"%(i,node_id))
'''