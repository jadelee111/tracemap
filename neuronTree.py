#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 12:16:03 2018

@author: chen
"""

import numpy as np
import copy
import importlib as imp
from sklearn.metrics.pairwise import cosine_similarity

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
        return max_idx[0]
    else:
        return (encodings,sphere_vectors[max_idx])

class Treenode():
    def __init__(self,node_id,node_type,x,y,z,radius,pid):
        self.node_id = node_id
        self.node_type = node_type
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.pid = pid
        self.children_id = []
        self.parent_encoding_max_id = -1  #compare with 48 sphere units, pointing from parent node to cur node, only one exists.
        self.children_encoding_max_id = []  #compre with 48 shpere units, pointing from cur node to children nodes. length the same to children_id
        
    def __str__(self):
        return "id: %d, pid: %d, x: %1f, y:%1f, z: %1f" % (self.node_id, self.pid,self.x,self.y,self.z)

        
        
class Neurontree():
    def __init__(self, nodes): #nodes is a list of Treenode
        self.neurontree = {}
        for i in range(len(nodes)):
            node_id = nodes[i].node_id
            self.neurontree[node_id]=nodes[i]
        #print("%d,%d"%(i,len(self.neurontree)))
        self.AssignParentEncodings()
        self.AssignChildren()
        
    def __len__(self):
        return len(self.neurontree)
        
    def GetNode(self,node_id):
        if node_id>0:
            return self.neurontree[node_id]
        else:
            return None
    
    def AssignChildren(self):
        for i,node_id in enumerate(self.neurontree):
            this_node = self.GetNode(node_id)
            parent_node = self.GetNode(this_node.pid)
            if not parent_node: #root node
                continue
            if node_id not in parent_node.children_id:
                parent_node.children_id.append(node_id)
                parent_node.children_encoding_max_id.append(this_node.parent_encoding_max_id)
                
    def AssignParentEncodings(self):
        for i,node_id in enumerate(self.neurontree):
            this_node = self.GetNode(node_id)
            x,y,z = this_node.x,this_node.y,this_node.z
            if this_node.pid>0:
                parent_node = self.GetNode(this_node.pid)
                px,py,pz = parent_node.x,parent_node.y,parent_node.z
                vector = np.array([[x-px,y-py,z-pz]])
                this_node.parent_encoding_max_id= return_one_hot_encoding(vector,True)
            
    def GetParentEncoding(self,node_id):
        max_id = self.neurontree[node_id].parent_encoding_max_id
        encodings =np.zeros((48,))
        if max_id>0:
            encodings[max_id] =1
        return encodings
        
    def GetChildrenEncodings(self,node_id):
        max_ids = self.neurontree[node_id].children_encoding_max_id
        encodings =np.zeros((48,))
        if max_ids:
            for i in max_ids:
                encodings[i]=1
        return encodings
    
    def GetChildrenNodes(self,node_id):        
        if node_id>0:
            children_nodes_id = self.neurontree[node_id].children_id
            if children_nodes_id: #if there are children
                node_list = [self.GetNode(i) for i in children_nodes_id]
                return node_list
        return []
    
    def GetChildrenNodes_multiple_steps(self,node_id, num_steps = 1):
        if num_steps==1:
            return self.GetChildrenNodes(node_id)
        # for num_steps>1
        step_count = 0
        seeds =[node_id]
        while step_count<num_steps:
            new_seed_ids= []
            #print('step:', step_count, ' seeds',seeds)
            while seeds:
                seed_id = seeds.pop()
                children_node_ids = self.GetChildrenNodes(seed_id)
                if children_node_ids:
                    new_seed_ids.extend([a.node_id for a in children_node_ids])
            seeds = copy.deepcopy(new_seed_ids)
            step_count+=1
        return [self.GetNode(i) for i in seeds]
    
    def HasChildren(self,node_id,num_steps = 1):
        res = self.GetChildrenNodes_multiple_steps(node_id,num_steps)
        if res:
            return True
        else:
            return False
        
    def HasParent(self,node_id):
        return self.neurontree[node_id].pid>0

    def GetChildrenNodesCoords(self,node_id,num_steps=1):
        #return the coords of children nodes
        nodes = self.GetChildrenNodes_multiple_steps(node_id,num_steps)
        if nodes:
            coords = []
            for i in range(len(nodes)):
                coords.append([nodes[i].x,nodes[i].y,nodes[i].z])
            return coords
        return None
            
    def GetAllNodeids(self):
        return list(self.neurontree.keys())
    
    def Copy(self):
        newNT=[]
        for i,node_id in enumerate(self.neurontree):
            n = self.neurontree[node_id]
            newNT.append(Treenode(n.node_id,n.node_type,n.x,n.y,n.z,n.radius,n.pid))
        return Neurontree(newNT)
        
    def Translate(self,offset_x,offset_y,offset_z):
        for i,node_id in enumerate(self.neurontree):
            n = self.neurontree[node_id]
            n.x -= offset_x
            n.y -= offset_y
            n.z -= offset_z

    def Save(self,filename,node_type =None):
        '''node_type color
        0: white
        1: black
        2: red
        3: blue
        4: purple
        5: green
        6 yellow
        7: another green
        '''
        file = open(filename,"w")
        file.write("##n,type,x,y,z,radius,parent\n")
        if node_type is not None:
            for i,node_id in enumerate(self.neurontree):
                node = self.neurontree[node_id]
                file.write("%d %d %4f %4f %4f %4f %d\n"%(node.node_id,node_type,
                                                     node.x,node.y, node.z,
                                                     node.radius,node.pid))            
        else:
            for i,node_id in enumerate(self.neurontree):
                node = self.neurontree[node_id]
                file.write("%d %d %4f %4f %4f %4f %d\n"%(node.node_id,node.node_type,
                                                     node.x,node.y, node.z,
                                                     node.radius,node.pid))
        file.close()














