from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import cv2
import argparse
import pdb
import tensorflow as tf
from skimage.transform import resize
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.spatial.distance import cdist
import sys
sys.path.insert(0, "/home/hxw/frameworks/models/research/object_detection")
from utils import visualization_utils as vis_util
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from itertools import compress
import matplotlib.pyplot as plt

def save_id_mapping(source_view, this_view, pid_mapping, bid_mapping):
    # save the pid/bid mapping.
    out_text = './result/reid/' + this_view + '_idmapping_source_'+ source_view + '.txt'
    f = open(out_text, 'w')
    tmp = [str(x) for x in pid_mapping]
    print(', '.join(tmp),file=f)
    tmp = [str(x) for x in bid_mapping]
    print(', '.join(tmp),file=f)
    f.close()    

def target_id_mapping(exp1_features, exp2_features, clss):
    dist_mat = cdist(exp1_features[:,2:], exp2_features[:,2:], 'cosine') 

    n_exp1 = len(np.unique(exp1_features[:,1]))
    n_exp2 = len(np.unique(exp2_features[:,1]))

    exp1_ids = exp1_features[:,1]
    exp2_ids = exp2_features[:,1]

    cls_dist_mat = np.ones([n_exp1, n_exp2]) * 1e8

    for i1 in range(n_exp1):
        for i2 in range(n_exp2):
            if abs(i1-i2) <= 4:
                l1 = sum(exp1_features[:,1] == i1) // 2
                l2 = sum(exp2_features[:,1] == i2) // 2
                cls_dist_mat[i1, i2] = np.mean(dist_mat[exp1_ids==i1][-l1:,exp2_ids==i2][:, :l2])       

    cam2_indices = np.array(range(cls_dist_mat.shape[-1])).astype(np.int32)  
    unmatched = cam2_indices
    matched = np.array([], dtype=np.int32)
    target_id_mapping = np.zeros(len(cam2_indices))

    MAXROUND=10
    i = 0
    while unmatched.any() and i<MAXROUND:
        indices = linear_assignment(cls_dist_mat[:, unmatched].T)
        matched = np.concatenate((matched, indices[:,0])) if matched.size else indices[:,0]
        target_id_mapping[unmatched[indices[:,0]]] = indices[:,1]
        unmatched = np.setdiff1d(cam2_indices, matched)
        i += 1     

    for i in unmatched:
        target_id_mapping[i] = np.argmin(cls_dist_mat[:, i]) 

    return target_id_mapping.astype(int)          

def reid(source_view, exps):
    exps = exps.split(",")

    person_features = {}
    bin_features = {}
    for exp in exps:
        person_features.update({exp: np.load('./result/tracking/' + exp + '_features_person.npy')})
        bin_features.update({exp: np.load('./result/tracking/' + exp + '_features_bin.npy')})

    # match source camera first
    # handle person only. we dont handle source bins
    print('>> creating source camera mapping %s' % (source_view))
    source_ids = person_features[source_view][:,1]
    source_dist_mat = cdist(person_features[source_view][:,2:], person_features[source_view][:,2:], 'cosine') 
    n_source = len(np.unique(source_ids))
    source_cls_dist_mat = np.zeros([n_source, n_source])

    for i1 in range(n_source):
        for i2 in range(n_source):
            source_cls_dist_mat[i1, i2] = np.mean(source_dist_mat[source_ids==i1][:,source_ids==i2])      
     
    THR = 0.07
    source_pid_mapping = np.array(range(n_source)).astype(int)
    
    queue = range(n_source)
    for i1 in queue:
        candidates = []
        for i2 in queue:
            if source_cls_dist_mat[i1, i2] <= THR and abs(i2-i1)<=2:
                candidates += [i2]
        for i2 in candidates:
            source_pid_mapping[i2] = i1
            queue.remove(i2)          
    
    source_bids = bin_features[source_view][:,1]
    n_source = len(np.unique(source_bids))
    source_bid_mapping = np.array(range(n_source)).astype(int)  
    save_id_mapping(source_view, source_view, source_pid_mapping, source_bid_mapping)


    target_views = set(exps) - set([source_view])
    
    for exp in target_views:
        print('>> creating target camera mapping %s --> %s' % (exp, source_view))
        if person_features[exp].size:
            pid_mapping = target_id_mapping(person_features[source_view], person_features[exp], 'person')
            target_pid_mapping = source_pid_mapping[pid_mapping]
        else:
            target_pid_mapping = []
        
        if bin_features[exp].size:
            bid_mapping = target_id_mapping(bin_features[source_view], bin_features[exp], 'bin')
            target_bid_mapping = source_bid_mapping[bid_mapping]
        else:
            target_bid_mapping = []    

        save_id_mapping(source_view, exp, target_pid_mapping, target_bid_mapping) 

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-Id")
    parser.add_argument(
        "--source_view", help="Name of Source Camera",
        default=None, required=True)
    parser.add_argument(
	    "--exps", help="Name of all cameras",
	    default=None, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reid(args.source_view, args.exps)