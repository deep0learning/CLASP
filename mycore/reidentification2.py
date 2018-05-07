#from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function
import re
import cv2
import argparse
import pdb
import tensorflow as tf
from skimage.transform import resize
from scipy.spatial.distance import cdist
from sklearn.utils.linear_assignment_ import linear_assignment
import sys
sys.path.insert(0, "/home/hxw/frameworks/models/research/object_detection")
from utils import visualization_utils as vis_util
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from itertools import compress
import matplotlib.pyplot as plt


def write_persons_on_image_array(image_np, image, boxes, ids):
    w = 960
    h = 540

    margin = int(10)

    sw = int((w - 6*margin)/5)
    sh = int((h - 4*margin)/3)

    image_np = np.array(image_np)
    txtcolor = 'red'
    try:
        font = ImageFont.truetype('demo/arial.ttf', 30)
    except IOError:
        font = ImageFont.load_default()

    #image = np.zeros([h, w, 3], dtype=np.uint8)
    for box, idx in zip(boxes, ids):
        (left, right, top, bottom) = (int(max(box[0],0)), int(min(box[0]+box[2], 1920)), int(max(box[1],0)), int(min(box[1]+box[3],1080)))
        subimage = image_np[top:bottom, left:right, :]

        newleft = margin*(int(idx)%5+1)+sw*(int(idx)%5)
        newright = newleft + sw
        #pdb.set_trace()
        newtop = margin*(int(idx)/5+1) + sh*(int(idx)/5)
        newbottom = newtop + sh

        newsubimage = Image.fromarray((resize(subimage, [sh, sw])*255).astype(np.uint8))
        
        draw = ImageDraw.Draw(newsubimage)
        display_str = 'Person %d' % (idx+1)
        text_width, text_height = font.getsize(display_str + '  ')
        text_bottom = 0 + text_height + 2*margin

        draw.rectangle([(0, text_bottom - text_height - 2 * margin), (0 + text_width, text_bottom)], fill='white')
        draw.text((0 + margin, text_bottom - text_height - margin), display_str, fill=txtcolor, font=font)

        #pdb.set_trace()

        image[newtop:newbottom, newleft:newright, :] = np.array(newsubimage).astype(np.uint8)

    return image

def write_tracks_on_image_array(image_pil, boxes, strs):
    for box, strg in zip(boxes, strs):
        write_tracks_on_image(image_pil, box, strg)
    image = np.array(image_pil)
    return image

def write_tracks_on_image(image, box, display_str):
    txtcolor = 'red'
    if 'bin' in display_str:
        boxcolor = 'green'
    else:
        boxcolor = 'blue'
    draw = ImageDraw.Draw(image)

    (left, right, top, bottom) = (box[0], box[0]+box[2], box[1], box[1]+box[3])
    try:
        font = ImageFont.truetype('demo/arial.ttf', 30)
    except IOError:
        font = ImageFont.load_default()
    
    text_width, text_height = font.getsize(display_str + '  ')
    margin = np.ceil(0.5 * text_height)

    text_bottom = top + text_height + 2*margin
    
    draw.rectangle(
         [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                           text_bottom)],
         fill='white')

    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=4, fill=boxcolor)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill=txtcolor,
        font=font)

def reid(exp1, exp2, vis=True):
    exp1_person_features_npy = './result/tracking/' + exp1 + '_features_person.npy'
    exp1_bin_features_npy = './result/tracking/' + exp1 + '_features_bin.npy'

    exp2_person_features_npy = './result/tracking/' + exp2 + '_features_person.npy'
    exp2_bin_features_npy = './result/tracking/' + exp2 + '_features_bin.npy'

    # match person
    exp1_person_features = np.load(exp1_person_features_npy)
    exp2_person_features = np.load(exp2_person_features_npy)

    dist_mat = cdist(exp1_person_features[:,2:], exp2_person_features[:,2:], 'cosine') 

    n_exp1 = len(np.unique(exp1_person_features[:,1]))
    n_exp2 = len(np.unique(exp2_person_features[:,1]))

    exp1_ids = exp1_person_features[:,1]
    exp2_ids = exp2_person_features[:,1]

    cls_dist_mat = np.zeros([n_exp1, n_exp2])


    for i1 in range(n_exp1):
        for i2 in range(n_exp2):
            cls_dist_mat[i1, i2] = np.mean(dist_mat[exp1_ids==i1][:,exp2_ids==i2])

    cam2_indices = np.array(range(cls_dist_mat.shape[-1])).astype(np.int32)  
    unmatched = cam2_indices
    matched = np.array([], dtype=np.int32)
    idx2_mapping = np.zeros(len(cam2_indices))

    MAXROUND=10
    i = 0

    pdb.set_trace()

    while unmatched.any() and i<MAXROUND:
        indices = linear_assignment(cls_dist_mat[:, unmatched].T)
        matched = np.concatenate((matched, indices[:,0])) if matched.size else indices[:,0]
        idx2_mapping[unmatched[indices[:,0]]] = indices[:,1]
        unmatched = np.setdiff1d(cam2_indices, matched)
        i += 1

    for i in unmatched:
        idx2_mapping[i] = np.argmin(cls_dist_mat[:, i])        

    # idx2_mapping = np.zeros(n_exp2)     
    # for i2 in range(n_exp2):
    #     idx2_mapping[i2] = np.argmin(cls_dist_mat[:, i2])     

    idx1_mapping = np.array(range(n_exp1))    

    # match bins 



    if vis:    

        i1 = 1
        i2 = 1

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./result/reid/' + exp1 + '_' + exp2 + '_XCamera_Matching.avi', fourcc, 50, (1920, 1080))
        
        # video 1
        video_file1 = "result/original/" + exp1 + ".mp4"
        person_track_file1 = 'result/tracking/' + exp1 + '_person.txt'
        bin_track_file1 = 'result/tracking/' + exp1 + '_bin.txt'

        capture1 = cv2.VideoCapture(video_file1)
        capture1.set(1, i1)

        person_boxes1 = np.loadtxt(person_track_file1, delimiter=',')
        bin_boxes1 = np.loadtxt(bin_track_file1, delimiter=',')

        
        # video 2
        video_file2 = "result/original/" + exp2 + ".mp4"
        person_track_file2 = 'result/tracking/' + exp2 + '_person.txt'
        bin_track_file2 = 'result/tracking/' + exp2 + '_bin.txt'

        capture2 = cv2.VideoCapture(video_file2)
        capture2.set(1, i2)

        person_boxes2 = np.loadtxt(person_track_file2, delimiter=',')
        bin_boxes2 = np.loadtxt(bin_track_file2, delimiter=',')
        image_np_person1 = np.zeros([540,960,3], dtype=np.uint8)
        image_np_person2 = np.zeros([540,960,3], dtype=np.uint8)


        while True:
            image_np_big = np.zeros([1080,1920,3], dtype=np.uint8)


            _, frame1 = capture1.read()
            if frame1 is not None:
                #pdb.set_trace()
                frame1 = frame1[:,:,::-1]

                image1 = Image.fromarray(frame1)
                i1 += 1

                curr_person_boxes1 = person_boxes1[person_boxes1[:,0] == i1]
                curr_bin_boxes1    = bin_boxes1[bin_boxes1[:,0] == i1]

                curr_person_ids1 = idx1_mapping[curr_person_boxes1[:,1].astype(int)]
                curr_person_strs1 = ['person %d' % (idx + 1) for idx in curr_person_ids1]

                curr_bin_strs1 = ['bin' for idx in range(len(curr_bin_boxes1))]

                image_np_person1 = write_persons_on_image_array(
                    image1, image_np_person1,
                    curr_person_boxes1[:,2:],
                    curr_person_ids1)

                image_np1 = write_tracks_on_image_array(
                    image1,
                    curr_person_boxes1[:,2:],
                    curr_person_strs1)

                image_np1 = write_tracks_on_image_array(
                    image1,
                    curr_bin_boxes1[:,2:],
                    curr_bin_strs1)

                image_np_big[0:540, 0:960, :] = (resize(image_np1, (540, 960))*255).astype(np.uint8)
                image_np_big[0:540, 960:, :] = image_np_person1


            _, frame2 = capture2.read()
            if frame2 is not None:
                #pdb.set_trace()
                frame2 = frame2[:,:,::-1]
                image2 = Image.fromarray(frame2)
                i2 += 1

                curr_person_boxes2 = person_boxes2[person_boxes2[:,0] == i2]
                curr_bin_boxes2    = bin_boxes2[bin_boxes2[:,0] == i2]

                curr_person_ids2 = idx2_mapping[curr_person_boxes2[:,1].astype(int)]
                curr_person_strs2 = ['person %d' % (idx + 1) for idx in curr_person_ids2]

                curr_bin_strs2 = ['bin' for idx in range(len(curr_bin_boxes2))]

                image_np_person2 = write_persons_on_image_array(
                    image2, image_np_person2,
                    curr_person_boxes2[:,2:],
                    curr_person_ids2)

                image_np2 = write_tracks_on_image_array(
                    image2,
                    curr_person_boxes2[:,2:],
                    curr_person_strs2)

                image_np2 = write_tracks_on_image_array(
                    image2,
                    curr_bin_boxes2[:,2:],
                    curr_bin_strs2)

                image_np_big[540:, 0:960, :] = (resize(image_np2, (540, 960))*255).astype(np.uint8)
                image_np_big[540:, 960:, :] = image_np_person2


            if frame1 is None and frame2 is None:
                break    
            
            

            image_np_big = image_np_big[:,:,::-1]
            out.write(image_np_big)
            print('%d frames processed!' % (i1))
        capture1.release()
        capture2.release()
        out.release()
        cv2.destroyAllWindows()    


        

    

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-Id")
    parser.add_argument(
        "--exp1", help="Name of Camera 1",
        default=None, required=True)
    parser.add_argument(
	    "--exp2", help="Name of Camera 2",
	    default=None, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reid(args.exp1, args.exp2)