#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:07:13 2018

@author: chen
"""
import data_utils as utils
from generator import data_generator_undirected,data_generator_directed,sample_nodes_truth

#swc_filename= '../data/train/140918c8/140918c8.tif.v3dpbd.swc'
#img_filename = '../data/train/140918c8/140918c8.tif'
#nt = utils.read_swc_get_nt(swc_filename)
#nt.HasChildren(1014)
train_dir ='../data/train'
traindatalist = utils.image_ids_in(train_dir)
traindatalist2 = [traindatalist[7]]
#
for x,y in data_generator_directed(train_dir,traindatalist2):
    pass

#a,b = sample_nodes_truth(swc_filename,img_filename,num_nodes_per_img=10, 
#                       child_step =1, side_length = 10, vis_enlarge_ratio =5)

