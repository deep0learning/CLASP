from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import cv2
import argparse
import pdb
import tensorflow as tf
from skimage.transform import resize
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
    exp1_person_features_npy = './demo/' + exp1 + '_features_person.npy'
    exp1_bin_features_npy = './demo/' + exp1 + '_features_bin.npy'

    exp2_person_features_npy = './demo/' + exp2 + '_features_person.npy'
    exp2_bin_features_npy = './demo/' + exp2 + '_features_bin.npy'

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

    idx2_mapping = np.zeros(n_exp2)     
    for i2 in range(n_exp2):
        idx2_mapping[i2] = np.argmin(cls_dist_mat[:, i2])  

    idx1_mapping = np.array(range(n_exp1))    

    # match bins 



    if vis:    
        for exp, mapping in zip([exp1, exp2], [idx1_mapping, idx2_mapping]):
            i = 0
            video_file = "demo/" + exp + ".mp4"
            person_track_file = 'demo/' + exp + '_person.txt'
            bin_track_file = 'demo/' + exp + '_bin.txt'

            capture = cv2.VideoCapture(video_file)
            capture.set(1, i)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('./demo/' + exp + '_XCamera_Matching.avi', fourcc, 50, (1920, 1080))

            person_boxes = np.loadtxt(person_track_file, delimiter=',')
            #bin_boxes = np.loadtxt(bin_track_file, delimiter=',')

            while True:
                flag, frame = capture.read()
                if frame is not None:
                    #pdb.set_trace()
                    frame = frame[:,:,::-1]
                    image = Image.fromarray(frame)
                    i += 1
                else: break
                
                curr_person_boxes = person_boxes[person_boxes[:,0] == i]
                #curr_bin_boxes    = bin_boxes[bin_boxes[:,0] == i]

                curr_person_ids = mapping[curr_person_boxes[:,1].astype(int)]
                curr_person_strs = ['person %d' % (idx + 1) for idx in curr_person_ids]

                image_np = write_tracks_on_image_array(
                    image,
                    curr_person_boxes[:,2:],
                    curr_person_strs)

                # image_np = write_tracks_on_image_array(
                #     image,
                #     curr_bin_boxes[:,2:],
                #     curr_bin_boxes[:,1],
                #     bin_ass,
                #     'bin')

                image_np = image_np[:,:,::-1]
                out.write(image_np)
                print('%d frames processed!' % (i))
            capture.release()
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