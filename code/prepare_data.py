#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 11:37:35 2018

@author: chen
"""
from data_utils import *
from neuronTree import Treenode,Neurontree
import random

filename = '../data/cd1.swc'
#filename = '../data/test.swc'
swc = read_swc(filename)
recon_list =[]
for i in range(len(swc)):
    one_node = Treenode(swc['id'][i],swc['type'][i],swc['x'][i],swc['y'][i],swc['z'][i],swc['r'][i],swc['pid'][i])
    recon_list.append(one_node)
    
Nt = Neurontree(recon_list)
sphere_vectors = load_sphere_48_units()

img_filename ='../data/cd1.tif'
img = load_tiff(img_filename)
side_length = 10
enlarged_ratio =5

node_ids = Nt.GetAllNodeids()
sel_ids = random.sample(node_ids,10)

for node_id in sel_ids:
    out_swc = []
    cur_node  = Nt.GetNode(node_id)
    #get the cropped image with the center of cur_node
    crop_img,xmin,ymin,zmin = read_cropped_image(cur_node.x,cur_node.y,cur_node.z,img,side_length =side_length)
    #generate the transformed whole image swc 
    trans_swc = swc_xyztransform(recon_list,xmin,ymin,zmin)
    center_node = Treenode(1,1,cur_node.x-xmin,cur_node.y-ymin,cur_node.z-zmin,0,-1)
    children_node_coord = Nt.GetChildrenNodesCoords(node_id)
    out_swc.append(center_node)
    cur_node_coord = np.array([cur_node.x,cur_node.y,cur_node.z])
    vectors = children_node_coord - cur_node_coord
    encoding,max_idx = return_one_hot_encoding(vectors)
    unit_vectors= sphere_vectors[max_idx]
    for j in range(len(max_idx)):
        enlarged_vector = unit_vectors[j]*enlarged_ratio + [cur_node.x-xmin,cur_node.y-ymin,cur_node.z-zmin]
        node = Treenode(j+2,3,enlarged_vector[0],enlarged_vector[1],enlarged_vector[2],0,1)
        out_swc.append(node)
    
    #save results
    fn_allswc = '../results/transefered%d.swc'%node_id
    fn_traceswc = '../results/trace%d.swc'%node_id
    fn_cropimg = '../results/crop%d.tif'%node_id
    fn_anno = '../results/%d.ano'%node_id
    write_swc(fn_allswc,trans_swc)
    write_swc(fn_traceswc,out_swc)
    save_tiff(fn_cropimg,crop_img.astype('uint8'))
    write_ano_file(fn_cropimg, [fn_allswc,fn_traceswc],fn_anno)
