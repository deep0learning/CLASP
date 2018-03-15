from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import cv2
import pdb
import tensorflow as tf
from skimage.transform import resize
import sys
sys.path.insert(0, "/home/hxw/frameworks/models/research/object_detection")
from utils import visualization_utils as vis_util
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from itertools import compress


def write_tracks_on_image_array(image_pil, boxes, ids, ass, clss):
    for box, i in zip(boxes, ids):
        write_tracks_on_image(image_pil, box, i, ass, clss)
    image = np.array(image_pil)
    return image

def write_tracks_on_image(image, box, id, ass, clss):
    txtcolor = 'red'
    if clss == 'person':
        boxcolor = 'blue'
        display_str = '%s %d ' % (clss, int(id) + 1)
    else:
        boxcolor = 'green'
        cass = np.zeros_like(ass)
        for i in np.unique(ass):
            subarray = np.where(ass == i)
            cass[subarray[0]] = range(len(subarray[0]))       
        display_str = 'person %d, bin %d '%(ass[int(id)] + 1, cass[int(id)] + 1)
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

def time_matching(person_boxes, bin_boxes):
    pids = np.unique(person_boxes[:,1])
    bids = np.unique(bin_boxes[:,1])
    sim_mat = np.ones((len(pids), len(bids))) * 1e-8
    for p in pids:
        minp = min(person_boxes[person_boxes[:,1]==p, 0])
        maxp = max(person_boxes[person_boxes[:,1]==p, 0])
        for b in bids:
            minb = min(bin_boxes[bin_boxes[:,1]==b, 0])
            maxb = max(bin_boxes[bin_boxes[:,1]==b, 0])
            tmp = min(maxp, maxb) - max(minp, minb)
            sim_mat[int(p), int(b)] = np.maximum(0, tmp)
    return sim_mat

def location_matching(person_boxes, bin_boxes):
    pids = np.unique(person_boxes[:,1])
    bids = np.unique(bin_boxes[:,1])
    dist_mat = np.ones((len(pids), len(bids))) * 1e8
    for p in pids:
        curr_person_boxes = person_boxes[person_boxes[:,1]==p]
        minp = min(curr_person_boxes[:,0])
        maxp = max(curr_person_boxes[:,0])
        for b in bids:
            curr_bin_boxes = bin_boxes[bin_boxes[:,1]==b]
            minb = min(curr_bin_boxes[:,0])
            maxb = max(curr_bin_boxes[:,0])
            if min(maxp, maxb) > max(minp, minb):
                acc = 0
                acc_d = 0
                for fr in range(int(max(minp, minb)), int(min(maxp, maxb))):
                    xy_p = curr_person_boxes[curr_person_boxes[:,0]==fr, 2:4] + curr_person_boxes[curr_person_boxes[:,0]==fr, 4:6]
                    xy_b = curr_bin_boxes[curr_bin_boxes[:,0]==fr, 2:4] + curr_bin_boxes[curr_bin_boxes[:,0]==fr, 4:6]
                    if xy_p.size > 0 and xy_b.size > 0:
                        acc_d += np.linalg.norm(xy_p - xy_b)
                        acc += 1
                if acc > 0:
                    avg_d = acc_d / acc
                    dist_mat[int(p), int(b)] = avg_d
    return dist_mat    


def associate(exp, startf=0, endf=100000, vis=True, fps=20.0):
    video_file = "demo/" + exp + ".mp4"
    person_track_file = 'demo/' + exp + '_person.txt'
    bin_track_file = 'demo/' + exp + '_bin.txt'

    capture = cv2.VideoCapture(video_file)
    capture.set(1, startf)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./demo/' + exp + '_SORT_Tracking.avi', fourcc, fps, (1920, 1080))

    person_boxes = np.loadtxt(person_track_file, delimiter=',')
    bin_boxes = np.loadtxt(bin_track_file, delimiter=',')


    time_sim = time_matching(person_boxes, bin_boxes)
    location_dist = location_matching(person_boxes, bin_boxes)

    co_dist = np.divide(location_dist, time_sim)

    bin_ass = np.argmin(co_dist, axis=0)

    i = startf - 1

    if vis:
        while True:
            flag, frame = capture.read()
            if frame is not None:
                #pdb.set_trace()
                frame = frame[:,:,::-1]
                image = Image.fromarray(frame)
                i += 1
            else: break
            if i > endf: break
            
            curr_person_boxes = person_boxes[person_boxes[:,0] == i]
            curr_bin_boxes    = bin_boxes[bin_boxes[:,0] == i]


            image_np = write_tracks_on_image_array(
                image,
                curr_person_boxes[:,2:],
                curr_person_boxes[:,1],
                bin_ass,
                'person')

            image_np = write_tracks_on_image_array(
                image,
                curr_bin_boxes[:,2:],
                curr_bin_boxes[:,1],
                bin_ass,
                'bin')

            image_np = image_np[:,:,::-1]
            out.write(image_np)
            print('%d frames processed!' % (i - startf + 1))
  
