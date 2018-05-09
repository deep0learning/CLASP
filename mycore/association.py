from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import cv2
import pdb
import tensorflow as tf
from datetime import datetime
from skimage.transform import resize
from sklearn.utils.linear_assignment_ import linear_assignment
import sys
sys.path.insert(0, "/home/hxw/frameworks/models/research/object_detection")
from utils import visualization_utils as vis_util
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from itertools import compress
from collections import Counter
import argparse
import os

def box_transform(rbbox, image_size):
    (h, w) = image_size
    xmin = rbbox[1]
    ymin = rbbox[0]
    xmax = rbbox[3]
    ymax = rbbox[2]

    (left, right, top, bottom) = (xmin * w, xmax * w,
                                  ymin * h, ymax * h) 
    width = right - left
    height = bottom - top
    return [left, top, width, height]

def write_event_on_image(image, frame, bin_person, event_frames):
    display_str = ''
    
    for b in event_frames:
        for fr in event_frames[b]:
            diff = frame - fr
            if 0 < diff < 30:
                person = bin_person[b]
                display_str += 'Person %d drops into Bin %d - \n' % (person, b)

    txtcolor = 'red'
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('demo/arial.ttf', 60)
    draw.text(
        (5, 5),
        display_str,
        fill=txtcolor,
        font=font)

    return np.array(image)

def write_tracks_on_image_array(image_pil, boxes, ids, ass, clss):
    for box, i in zip(boxes, ids):
        write_tracks_on_image(image_pil, box, i, ass, clss)
    image = np.array(image_pil)
    return image

def write_tracks_on_image(image, box, id, ass, clss):
    txtcolor = 'red'
    if clss == 'person':
        boxcolor = 'blue'
        display_str = '%s %d ' % (clss, int(id))
    elif clss == 'hand':
        boxcolor = 'red'
        display_str = '%s' % (clss)
        box = box_transform(box, [1080, 1920])
    else:
        boxcolor = 'green'
        #cass = np.zeros_like(ass)
        # for i in np.unique(ass):
        #     subarray = np.where(ass == i)
            #cass[subarray[0]] = range(len(subarray[0]))       
        display_str = 'person %d, bin %d'%(ass[int(id)], id)#cass[int(id)] + 1)
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

def location_matching(person_boxes, bin_boxes):
    pids = np.unique(person_boxes[:,1])
    bids = np.unique(bin_boxes[:,1])
    dist_mat = np.ones((len(pids), len(bids))) * 1e8
    for p in range(len(pids)):
        curr_person_boxes = person_boxes[person_boxes[:,1]==pids[p]]
        minp = min(curr_person_boxes[:,0])
        maxp = max(curr_person_boxes[:,0])
        for b in range(len(bids)):
            curr_bin_boxes = bin_boxes[bin_boxes[:,1]==bids[b]]
            minb = min(curr_bin_boxes[:,0])
            maxb = max(curr_bin_boxes[:,0])
            if min(maxp, maxb) > max(minp, minb):
                acc = 0
                acc_d = 0
                for fr in range(int(max(minp, minb)), int(min(maxp, maxb))):
                    xy_p = curr_person_boxes[curr_person_boxes[:,0]==fr, 2:4] + curr_person_boxes[curr_person_boxes[:,0]==fr, 4:6]/2
                    xy_b = curr_bin_boxes[curr_bin_boxes[:,0]==fr, 2:4] + curr_bin_boxes[curr_bin_boxes[:,0]==fr, 4:6]/2
                    if xy_p.size > 0 and xy_b.size > 0:
                        acc_d += np.linalg.norm((xy_p - xy_b))
                        acc += 1
                if acc > 0:
                    avg_d = acc_d / (acc)
                    dist_mat[int(p), int(b)] = avg_d
    return dist_mat 

def detect_event(hand_boxes, bin_boxes):
    bids = np.unique(bin_boxes[:,1]).astype(int)
    #event_frames = np.zeros_like(bids)
    event_frames = {}
    zero_pad = np.zeros_like(bin_boxes[:,0])
    bin_boxes = np.concatenate((bin_boxes, zero_pad[:,np.newaxis]), axis=1)

    for hand_box in hand_boxes:
        fr = hand_box[0]
        current_bin_boxes = bin_boxes[bin_boxes[:,0] == fr]

        for bin_box in current_bin_boxes:
            hbbox = box_transform(hand_box[2:], [1080, 1920])

            if hbbox[0] > bin_box[2] and hbbox[1] > bin_box[3] and (hbbox[0]+hbbox[2])<(bin_box[2]+bin_box[4]) and (hbbox[1]+hbbox[3])<(bin_box[3]+bin_box[5]):
               bin_box[-1] = 1   
        
        if current_bin_boxes.size > 0:    
            bin_boxes[bin_boxes[:,0] == fr] = current_bin_boxes

    for b in bids:
        event_frames[b] = []
        current_bin_boxes = bin_boxes[bin_boxes[:,1] == b]
        # for i in range(1, len(current_bin_boxes)):
        #     if current_bin_boxes[i][-1] == 1 and current_bin_boxes[i-1][-1] == 0:
        #         event_frames[b] += [current_bin_boxes[i][0]]  
        valid_bin_boxes = current_bin_boxes[current_bin_boxes[:,-1] == 1]
        if valid_bin_boxes.size > 0:
            event_frames[b] += [min(valid_bin_boxes[:,0])]        

    return event_frames

def write_loc_log(file, clss, camid, fr, bb, id, pid, first):
    log_str = 'LOC: type: %s camera-num: %d frame: %d time-offset: %d bb: %d %d %d %d ID: %d PAX-ID: %s first-used: %s ' % (clss, camid, int(fr), int(fr*33.3333), int(bb[0]), int(bb[1]), int(bb[0]+bb[2]), int(bb[1]+bb[3]), id, 'NA' if clss=='PAX' else str(int(pid)), str(first))
    print(log_str, file=file)           

def write_tr_log(file, camid, fr, bb, id, pid, fromto, theft):    
    log_str = 'XFR: type: %s camera-num: %d frame: %d time-offset: %d owner-ID: %d DVI-ID: %d theft: %s' % (fromto, camid, int(fr), int(fr*33.3333), pid, id, str(theft))
    print(log_str, file=file) 


def associate(exps, source, startf=0, endf=100000, vis=False, fps=45.0):
    # write log
    log_file = 'result/scoringTool/ATA_log_v3.txt'
    if os.path.isfile(log_file):
        os.remove(log_file)
    f = open(log_file, 'a')

    exp2cam = {'cam9exp5a': 9, 'cam11exp5a': 11, 'cam13exp5a': 13, 'cam2exp5a': 2, 'cam5exp5a': 5, 'cam9exp5b': 9, 'cam11exp5b': 11, 'cam13exp5b': 13, 'cam2exp5b': 2, 'cam5exp5b': 5}
    exp2type = {'cam9exp5a': 'to', 'cam2exp5a': 'to', 'cam5exp5a': 'from', 'cam11exp5a': 'from', 'cam13exp5a': 'from', 'cam9exp5b': 'to', 'cam2exp5b': 'to', 'cam5exp5b': 'from', 'cam11exp5b': 'from', 'cam13exp5b': 'from'}

    exps = exps.split(",")

    person_boxes = {}
    bin_boxes    = {}
    hand_boxes   = {}
    bin_person   = {}
    event_frames = {}
    first_person = {}
    first_bin    = {}

    for exp in exps:
        print('>> associationg exp %s ...' % exp)
        person_track_file = 'result/tracking/' + exp + '_person.txt'
        bin_track_file = 'result/tracking/' + exp + '_bin.txt'
        idmapping_file = 'result/reid/' + exp + '_idmapping_source_%s.txt' % source
        hand_det_file = 'result/detection/' + exp + '_FRCNN_DET_hand.npy'

        with open(idmapping_file) as idfile:
            content = idfile.readlines()  
        
        if os.path.isfile(person_track_file):
            person_boxes[exp] = np.loadtxt(person_track_file, delimiter=',')
            pid_mapping = np.array(content[0].split(', ')).astype(int)
            person_boxes[exp][:,1] = pid_mapping[person_boxes[exp][:,1].astype(int)]
            pids = np.unique(person_boxes[exp][:,1]).astype(int)
        else:
            person_boxes[exp] = np.array([])
            pids = np.array([])

        if os.path.isfile(bin_track_file):    
            bin_boxes[exp] = np.loadtxt(bin_track_file, delimiter=',')
            bid_mapping = np.array(content[1].split(', ')).astype(int)
            bin_boxes[exp][:,1] = bid_mapping[bin_boxes[exp][:,1].astype(int)]
            bids = np.unique(bin_boxes[exp][:,1]).astype(int)
        else:
            bin_boxes[exp] = np.array([])
            bids = np.array([])

        hand_boxes[exp] = np.load(hand_det_file)[:,:6]
        
        if person_boxes[exp].size and bin_boxes[exp].size:
            location_dist = location_matching(person_boxes[exp], bin_boxes[exp]) 

            bin_indices = np.array(range(len(bids)))  
            unmatched = bin_indices
            matched = np.array([], dtype=np.int32)
            bin_ass = np.zeros(len(bin_indices))

            MAXROUND=10
            i = 0

            while unmatched.any() and i<MAXROUND:
                indices = linear_assignment(location_dist[:, unmatched].T)
                matched = np.concatenate((matched, indices[:,0])) if matched.size else indices[:,0]
                bin_ass[unmatched[indices[:,0]]] = indices[:,1]
                unmatched = np.setdiff1d(bin_indices, matched)
                i += 1

            for i in unmatched:
                bin_ass[i] = np.argmin(location_dist[:, i]) 

            bin_person[exp] = {}
            for bid in bids:
                bin_person[exp][bid] = pids[int(bin_ass[int(np.argwhere(bids==bid))])]   
        
        if hand_boxes[exp].size and bin_boxes[exp].size:
            event_frames[exp] = detect_event(hand_boxes[exp], bin_boxes[exp])  
        else:
            event_frames[exp] = {}    

        for bid in bids:
            if bid in first_bin:
                first_bin[bid] = min(first_bin[bid], min(bin_boxes[exp][:,0]))
            else:
                first_bin[bid] = min(bin_boxes[exp][:,0]) 

        for pid in pids:
            if pid in first_person:
                first_person[pid] = min(first_person[pid], min(person_boxes[exp][:,0]))
            else:
                first_person[pid] = min(person_boxes[exp][:,0])               


    for exp in exps:
        camid = exp2cam[exp]

        for fr in range(0, int(max(max(person_boxes[exp][:,0]), max(bin_boxes[exp][:,0]) if bin_boxes[exp].size else 0))):
            # check loc person
            if person_boxes[exp].size:
                pboxes = person_boxes[exp][person_boxes[exp][:,0]==fr]

            if pboxes.size:
                for pbox in pboxes:
                    write_loc_log(f, 'PAX',  camid, fr, pbox[2:6], pbox[1], pbox[1], first_person[pbox[1]]==fr)        

            # check loc bin
            if bin_boxes[exp].size:
                bboxes = bin_boxes[exp][bin_boxes[exp][:,0]==fr]

            if bboxes.size:
                for bbox in bboxes:
                    write_loc_log(f, 'DVI',  camid, fr, bbox[2:6], bbox[1], bin_person[exp][bbox[1]], first_bin[bbox[1]]==fr)                

            # check transfer
            bboxids = [b for b in event_frames[exp] if fr in event_frames[exp][b]]
            if bboxids and fr != 0:
                for bboxid in bboxids: 
                    bbox = bboxes[bboxes[:,1]==bboxid]
                    if bbox.size:
                        bbox = bbox[0]
                        write_tr_log(f, camid, fr, bbox[2:6], bbox[1], bin_person[exp][bbox[1]], exp2type[exp], bin_person[exp][bbox[1]]==bin_person[source][bbox[1]])      
        
        if vis:
            video_file = "result/original/" + exp + ".mp4"
            capture = cv2.VideoCapture(video_file)
            capture.set(1, startf)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            i = startf - 1
            out = cv2.VideoWriter('./result/associate/' + exp + '_SORT_Tracking.avi', fourcc, fps, (1920, 1080))
            while True:
                flag, frame = capture.read()
                if frame is not None:
                    frame = frame[:,:,::-1]
                    image = Image.fromarray(frame)
                    i += 1
                else: break
                if i > endf: break

                curr_person_boxes = person_boxes[exp][person_boxes[exp][:,0] == i]
                if bin_boxes[exp].size:
                    curr_bin_boxes    = bin_boxes[exp][bin_boxes[exp][:,0] == i]
                curr_hand_boxes   = hand_boxes[exp][hand_boxes[exp][:,0] == i]

                image_np = write_tracks_on_image_array(
                    image,
                    curr_person_boxes[:,2:],
                    curr_person_boxes[:,1],
                    bin_person[exp],
                    'person')
                
                if curr_bin_boxes.size:
                    image_np = write_tracks_on_image_array(
                        image,
                        curr_bin_boxes[:,2:],
                        curr_bin_boxes[:,1],
                        bin_person[exp],
                        'bin')

                image_np = write_tracks_on_image_array(
                    image,
                    curr_hand_boxes[:,2:],
                    curr_hand_boxes[:,1],
                    [],
                    'hand')
                
                if len(bin_person[exp].keys())>0 and len(event_frames[exp].keys())>0:
                    image_np = write_event_on_image(
                        image,
                        i,
                        bin_person[exp],
                        event_frames[exp])

                image_np = image_np[:,:,::-1]
                out.write(image_np)
                print('%d frames processed!' % (i - startf + 1))


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Association")
    parser.add_argument(
        "--exps", help="Name of all cameras",
        default=None, required=True)
    parser.add_argument(
        "--source_view", help="Name of source camera",
        default=None, required=True)
    parser.add_argument(
        "--vis", help="Generate videos or not",
        default=False, required=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    associate(args.exps, args.source_view, vis=(args.vis=='True'))           
