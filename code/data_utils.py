#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:26:58 2018

@author: chen
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from libtiff import TIFFfile
from libtiff import TIFFimage
from neuronTree import Treenode, Neurontree
import os

def image_ids_in(root_dir):
    ids = []
    for id in os.listdir(root_dir):
        ids.append(id)
    return ids

def read_swc(filename):
    #output: a structured array with 7 fields
    swc = np.loadtxt(filename,dtype={'names': ('id','type', 'x', 'y','z','r','pid'),
                     'formats': ('i4','i4', 'f4','f4','f4','f4','i4')})
    return swc
    
def write_swc(filename,treenodelist,node_type= None):
    #treenodelist is a list of treenodes
    
    file = open(filename,"w")
    file.write("##n,type,x,y,z,radius,parent\n")
    if node_type is not None:
        for i in range(len(treenodelist)):
            file.write("%d %d %4f %4f %4f %4f %d\n"%(treenodelist[i].node_id,
                                                     node_type,
                                                     treenodelist[i].x,treenodelist[i].y,treenodelist[i].z,
                                                     treenodelist[i].radius,treenodelist[i].pid))
    else:
        for i in range(len(treenodelist)):
            file.write("%d %d %4f %4f %4f %4f %d\n"%(treenodelist[i].node_id,
                                                     treenodelist[i].node_type,
                                                     treenodelist[i].x,treenodelist[i].y,treenodelist[i].z,
                                                     treenodelist[i].radius,treenodelist[i].pid))
    file.close()
    
    
def read_swc_get_nt(filename):
    ''' read in a swc and return a neuron tree with children assigned'''
    swc = read_swc(filename)
    recon_list =[]
    for i in range(len(swc)):
        one_node = Treenode(swc['id'][i],swc['type'][i],swc['x'][i],swc['y'][i],swc['z'][i],swc['r'][i],swc['pid'][i])
        recon_list.append(one_node)
    #print('recon len:',len(recon_list))
    Nt = Neurontree(recon_list)

    return Nt

def swc_xyztransform(treenode_list,offset_x,offset_y,offset_z):
    '''
    input: treenode_lsit is a list of tree nodes
    offset_x,y,z are the offset used for the transformation
    '''
    backuplist = []
    for i in range(len(treenode_list)):
        tn = treenode_list[i]
        new_node = Treenode(tn.node_id,tn.node_type,tn.x-offset_x,tn.y-offset_y,tn.z-offset_z,tn.radius,tn.pid)
        backuplist.append(new_node)
    return backuplist


def load_sphere_48_units():
    table = np.loadtxt('../data/nhtmap_0.vtk', skiprows=5)    
    return table

def return_one_hot_encoding(xyzVectors,return_idx_flag = False):
    '''xyzVectors: the coord of childrens' nodes, needs to be an array of N sample *3
       scores are in the size N sample *48
       output: one hot encoding considering all childrens nodes. Encoding should be 48 in length
    '''
    sphere_vectors = load_sphere_48_units()
    scores = cosine_similarity(xyzVectors,sphere_vectors)
    max_idx = np.argmax(scores,axis=1)
    encodings =np.zeros((48,))
    encodings[max_idx] =1
    if return_idx_flag:
        return max_idx
    else:
        return (encodings,sphere_vectors[max_idx])

def calc_one_hot_encoding_wrapper(nt,node_id,num_steps=1):
    '''input
           nt -- neurontree
           node_id -- which node to calc
           num_steps -- how many steps to take to find the children. If =2, look at grandchildren
       output
           one-hot encoding of the node and the unit vectors. (unit vector could be a list)
    Note: in generator, we check whether this node has children first before compute one-hot encoding
    '''
    children_node_coord = nt.GetChildrenNodesCoords(node_id,num_steps)
    if not children_node_coord: #if this node does not have children
        return (None, None)
    cur_node  = nt.GetNode(node_id)
    cur_node_coord = np.array([cur_node.x,cur_node.y,cur_node.z])
    vectors = np.array(children_node_coord) - cur_node_coord
    return return_one_hot_encoding(vectors)


def load_tiff(fname):  #it reads in the order of z,y,xs
    #print(fname)
    tiff = TIFFfile(fname)
    #samples, sample_names = tiff.get_samples()
    arr = tiff.get_tiff_array(sample_index=0, subfile_type=0)
    arr= np.asarray(arr)
    arr1 = np.transpose(arr,(2,1,0))
    return arr1


def save_tiff(fname, arr, compression=None):
    arr1 = np.transpose(arr,(2,1,0))
    arr1 = arr1[:,::-1,:]
    tiff = TIFFimage(arr1)
    tiff.write_file(fname)
    
def write_ano_file(img_path, swc_paths,output_name):
    '''
    input: swc_paths could be several paths in a list
    '''
    file = open(output_name,"w")
    file.write("GRAYIMG=%s\n"%img_path)
    for i in range(len(swc_paths)):
        file.write("SWCFILE=%s\n"%swc_paths[i])
    file.close()
    
def accuracy(y_true,y_pred):
    y_pred[y_pred>0.5]=1
    y_pred[y_pred<0.5]=0
    intersection = np.sum(y_pred==y_true)
    return intersection/48.0
    
def read_cropped_image(x,y,z,volume,side_length = 10):
    '''
    xyz: the coords of the center of the crops, floats
    side_length: the side of the cropped cube
    output: cropped volume
    in the case where not enough space is left, we pad the volume with 0 voxels
    '''
    #need two coords. Coords for the big volume (xout_l, xout_r, yout_l,yout_r, zout_l, zout_r)
    #Coords for the crop volume (xcrop_l/r, ycropl/r, zcropl/r)
    x,y,z = int(round(x)),int(round(y)),int(round(z))
    h,w,d = volume.shape
    #print('hwd',h,w,d, ' xyz',x,y,z,'\n')
    side_total_length = side_length*2+1
    crop_volume = np.zeros((side_total_length,side_total_length,side_total_length))
    xmin, xmax = x-side_length, x+side_length
    ymin, ymax = y-side_length, y+side_length
    zmin, zmax = z-side_length, z+side_length
    xcrop_l, ycrop_l, zcrop_l = 0, 0, 0
    xcrop_r, ycrop_r, zcrop_r = side_total_length-1,side_total_length-1,side_total_length-1 #inclusive
    #crop_volume = volume[xmin:xmax,ymin:ymax,zmin:zmax]
    if xmin<0:
        xcrop_l = -xmin
        xmin = 0 
    if ymin<0:
        ycrop_l = -ymin
        ymin = 0
    if zmin<0:
        zcrop_l = -zmin
        zmin = 0
    if xmax>=h: #[xmin,xmax+1], xmax is inclusive
        xcrop_r = xcrop_r - (xmax-(h-1))
        xmax = h-1   
    if ymax>=w:
        ycrop_r = ycrop_r - (ymax-(w-1))
        ymax = w-1
    if zmax>=d:
        zcrop_r = zcrop_r - (zmax-(d-1))
        zmax = d-1
    
    assert xmax-xmin == xcrop_r-xcrop_l, "xcrop_r: %d, xcrop_l: %d, xmax: %d, xmin: %d" %(xcrop_r,xcrop_l,xmax,xmin)
    assert ymax-ymin == ycrop_r-ycrop_l, "xcrop_r: %d, xcrop_l: %d, xmax: %d, xmin: %d" %(ycrop_r,ycrop_l,ymax,ymin)
    assert zmax-zmin == zcrop_r-zcrop_l, "xcrop_r: %d, xcrop_l: %d, xmax: %d, xmin: %d" %(zcrop_r,zcrop_l,zmax,zmin)
    if zmax-zmin<=0 or xmax-xmin <=0 or ymax-ymin <=0:
        print('~~~~~~~error,x:%d y:%d  z:%d, h: %d, w: %d, d :%d'%(x,y,z,h,w,d))
    
    crop_volume[xcrop_l:(xcrop_r+1),ycrop_l:(ycrop_r+1),zcrop_l:(zcrop_r+1)] = volume[xmin:xmax+1,ymin:ymax+1,zmin:zmax+1]
    return (crop_volume, xmin,ymin,zmin)



        
        
    
    
    
    
    
    
    
    