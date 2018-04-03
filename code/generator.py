#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:20:41 2018

@author: chen
"""
import numpy as np
import random
import data_utils as utils
from neuronTree import Treenode, Neurontree

n_ch,n_x,n_y,n_z = 1, 24,24,24
sphere_vectors = utils.load_sphere_48_units()

def data_generator_undirected(folderpath,data_filelist, n_label= 48,batch_size= 20,num_nodes_per_img =50,child_step =1):
    '''
    read in a random image
    get 50 random node, extract cropped_img and the encoding

    '''
    
    count = 0
    batch = batch_size
    x_patch = np.zeros((batch,n_ch,n_x,n_y,n_z))
    y_patch = np.zeros((batch,n_label))

    while 1:
        data_file = random.choice(data_filelist)
        #print('%s\n',data_file)
        img_filename = folderpath + '/'+ data_file +'/'+ data_file +'.tif'
        swc_filename = folderpath + '/'+ data_file +'/'+data_file +'.tif.v3dpbd.swc'
        imgs,labels = sample_nodes_truth(swc_filename,img_filename,num_nodes_per_img=num_nodes_per_img, 
                                         child_step = child_step)
        #print('number of sampled img:',len(imgs))
        

        for i in range(len(imgs)):
            x_patch[count,0,2:-1,2:-1,2:-1] =imgs[i]
            y_patch[count,:] = labels[i]
            count+=1
#            fname='../data/patches/img'+str(count)+'.tif'
#            save_tiff(fname,patches[i])
#            fname='../data/patches/label'+str(count)+'.tif'
#            save_tiff(fname,plabels[i]*255)            

            if count==batch:
                yield(x_patch,y_patch)
                x_patch = np.zeros((batch,n_ch,n_x,n_y,n_z))
                y_patch = np.zeros((batch,n_label))
                count=0

def sample_nodes_truth(swc_filename,img_filename,num_nodes_per_img=10, 
                       child_step =1, side_length = 10, vis_enlarge_ratio =5):
    '''
    sample nodes from the loaded image
    note: it is possible the sampled image < num_nodes_per_img because some nodes do not have children
    '''
    #init output
    crop_img_list =list()
    encoding_list = list()
    #read img
    img = utils.load_tiff(img_filename)
    h,w,d = img.shape
    #read swc
    nt = utils.read_swc_get_nt(swc_filename)
    node_ids = nt.GetAllNodeids()
    #print('neurontree length: ',len(nt),'node_id length',len(node_ids))

    #sample nodes
    #sel_ids = random.sample(node_ids,num_nodes_per_img)
    sel_ids = [240,1014,1602,2583]
    #calc truth

    for node_id in sel_ids:
        #check if it has children
        if not nt.HasChildren(node_id,child_step): continue
        cur_node  = nt.GetNode(node_id)
        #check if the cur_node is within image
        if cur_node.x >= h or cur_node.y >=w or cur_node.z >=d: continue        
        
        #get the one_hot encoding
        encoding, unit_vectors = utils.calc_one_hot_encoding_wrapper(nt,node_id,num_steps = child_step)
        
        #get the cropped image with the center of cur_node
        crop_img,xmin,ymin,zmin = utils.read_cropped_image(cur_node.x,cur_node.y,cur_node.z,img,side_length =side_length)
        
        #append to list
        crop_img_list.append(crop_img)
        encoding_list.append(encoding)
        
        #For visualization
        #generate the transformed whole image swc
#        bk_nt = nt.Copy()
#        bk_nt.Translate(xmin,ymin,zmin)
#        out_swc = []
#        center_node = Treenode(1,2,cur_node.x-xmin,cur_node.y-ymin,cur_node.z-zmin,0,-1)
#        out_swc.append(center_node)
#        #the direction of unit vector
#        for j in range(len(unit_vectors)):
#            enlarged_vector = unit_vectors[j]*vis_enlarge_ratio + [center_node.x,center_node.y,center_node.z]
#            node = Treenode(j+2,2,enlarged_vector[0],enlarged_vector[1],enlarged_vector[2],0,1)
#            out_swc.append(node)
#        
#        #save results
#        fn_allswc = '../results/transefered%d_%d.swc'%(node_id,child_step)
#        fn_traceswc = '../results/trace%d_%d.swc'%(node_id,child_step)
#        fn_cropimg = '../results/crop%d_%d.tif'%(node_id,child_step)
#        fn_anno = '../results/%d_%d.ano'%(node_id,child_step)
#        bk_nt.Save(fn_allswc)
#        utils.write_swc(fn_traceswc,out_swc)
#        utils.save_tiff(fn_cropimg,crop_img.astype('uint8'))
#        utils.write_ano_file(fn_cropimg, [fn_allswc,fn_traceswc],fn_anno)
#    print('sample nodes', len(crop_img_list))
    return (crop_img_list,encoding_list)
        
        
    