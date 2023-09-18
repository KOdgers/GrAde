import os
import glob

import pickle as pkl
import numpy as np
import pandas as pd
import networkx as nx
import cv2
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/kelly/Documents/Projects/EntS')
from utils_ade20k import loadAde20K 
#from ADE.pregraph.utils.ade_to_graph import *
def load_ade_files(dpath, index_file,object_or_class = True):
    DATASET_PATH = dpath 
    index_file = index_file 
    with open('{}/{}/{}'.format(DATASET_PATH,'ADE20K_2021_17_01', index_file), 'rb') as f:
        index_ade20k = pkl.load(f)
    id_df = pd.DataFrame(zip(index_ade20k['objectnames'],index_ade20k['wordnet_level1']))
    id_df.columns=['object_name','class_name']
    return index_ade20k,id_df
def load_raw_subgraphs():
    sdict = pkl.load(open(f'../../impexp_subgraphs.pickle','rb'))
    return sdict

def get_data(dpaths):
    data,id_df = load_ade_files(dpaths['ade_root'],dpaths['ade_index'])
    return data,id_df 

def find_bbox(segments,cidx,class_mask, id_df):
    locs = np.argwhere(segments==cidx)
    x = np.unique(locs[:,1])
    y = np.unique(locs[:,0])
    # try:
    bbox = [min(x),max(x),min(y),max(y)]
    centers = [(bbox[0]+bbox[1])/2,(bbox[2]+bbox[3])/2]

    bbox = [(bbox[1]+bbox[0])/2,(bbox[3]+bbox[2])/2,bbox[1]-bbox[0], bbox[3]-bbox[2] ]
    
    class_inst = class_mask[locs[0,0],locs[0,1]]
    class_name = id_df.iloc[class_inst-1]['class_name']
    return bbox,class_name,centers

def parse_image_graph(img_name:str,datapaths:dict,id_df:pd.DataFrame):
    print(img_name)

    root_path = datapaths['ade_root']
    # root_path = '..'

    # print(folder1,folder2,filename)
#    full_file_name = '{}/{}'.format('images/ADE/training',img_name)
    full_file_name = img_name
    print(full_file_name)
    
    info = loadAde20K('{}/{}'.format(root_path, full_file_name))
    img = cv2.imread(info['img_name'])[:,:,::-1]

    ent = {}
    ent['segment']=info['instance_mask']
    ent['image'] = img
    temp_class_mask = info['class_mask']
    # width,height = info['class_mask'].shape

    ent['object'] = info['objects']['instancendx']
    objects = []

    bboxes = [[]]
    centers = [[]]
    for i in range(1,len(ent['object'])):
        bbox,class_inst,center_inst = find_bbox(ent['segment'],i,temp_class_mask,id_df)
        
        bboxes.append(bbox)
        centers.append(center_inst)
        class_idx = id_df[id_df.class_name==class_inst].index[0]
        # class_idx = id_df[id_df.class_name==class_inst].index[0]
        objects.append(class_idx)
    
    del bboxes[0] 
    del centers[0]

    ent['bbox'] =bboxes
    ent['center'] =centers
    ent['obj_idx'] = objects
    objects = [id_df.iloc[int(obj)].class_name for obj in objects]
    ent['objects'] = objects

    return  full_file_name,ent

def find_class(segments,cidx,class_mask, id_df):
    locs = np.argwhere(segments==cidx)

    class_inst = class_mask[locs[0,0],locs[0,1]]
    class_name = id_df.iloc[class_inst-1]['class_name']
    return class_name
def seg_to_lines(seg):
    vert = seg[:,:-1]-seg[:,1:]
    hort = seg[:-1,:]-seg[1:,:]
    lines = np.concatenate([np.where(vert!=0,1,0),np.zeros((vert.shape[0],1))],axis=1)
    lines_hort = np.concatenate([np.where(hort!=0,1,0), np.zeros((1,hort.shape[1]))], axis=0)
    lines = np.where(lines+lines_hort>0,1.,0.)
    for i in range(1,3):
        lines[i:,:] += lines[:-i,:]
        lines[:,i:]+= lines[:,:-i]
        lines[:-i,:] += lines[i:,:]
        lines[:,:-i]+= lines[:,i:]
    lines = np.where(lines>0,1.,0.)
    return lines 

    
    
def img_graph_subgraphs_meta(img_name:str,datapaths:dict,id_df:pd):
    fname,ent = parse_image_graph(img_name,datapaths,id_df)
    n_segments = np.max(ent['segment'])
    segment_lines = seg_to_lines(ent['segment'])
    seg_fig, seg_ax = plt.subplots(figsize=(10,10))
    seg_fig.subplots_adjust(0,0,1,1)
    seg_ax.imshow(cv2.cvtColor(ent['image'], cv2.COLOR_BGR2GRAY),cmap='gray')
    seg_ax.imshow(segment_lines,alpha=segment_lines)
    seg_ax.axis('off')
    seg_ax.margins(0,0)
#    plt.tight_layout()

    # seg_ax.imshow('segment'],alpha=0.3)
    plt.savefig('goshdarnit.png')
    return seg_fig, ent,segment_lines.shape

