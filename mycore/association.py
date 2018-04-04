from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import cv2
import pdb
import tensorflow as tf
from skimage.transform import resize
import sys
sys.path.insert(0, "/project/google-surv/alert/models/research/object_detection")
from utils import visualization_utils as vis_util
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from itertools import compress
from collections import Counter

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
    elif clss == 'hand':
        boxcolor = 'red'
        display_str = '%s %d ' % (clss, int(id) + 1)
    else:
        boxcolor = 'green'
        cass = np.zeros_like(ass)
        for i in np.unique(ass):
            subarray = np.where(ass == i)
            cass[subarray[0]] = range(len(subarray[0]))       
        display_str = 'person %d, bin %d ,bin %d'%(ass[int(id)] + 1, cass[int(id)] + 1, int(id))
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

def hand_matching(hand_boxes, person_boxes, bin_boxes):
    hids = np.unique(hand_boxes[:,1])
    pids = np.unique(person_boxes[:,1])
    bids = np.unique(bin_boxes[:,1])
    person_mat = np.ones((len(hids), len(pids))) * 1e-8
    bin_mat = np.ones((len(hids), int(max(bids)+1))) * 1e-8
    for h in hids:
        curr_hand_boxes = hand_boxes[hand_boxes[:,1]==h]
        minh = min(hand_boxes[hand_boxes[:,1]==h, 0])
        maxh = max(hand_boxes[hand_boxes[:,1]==h, 0])
        for p in pids:
            curr_person_boxes = person_boxes[person_boxes[:,1]==p]
            minp = min(curr_person_boxes[:,0])
            maxp = max(curr_person_boxes[:,0])
            if min(maxp, maxh) > max(minp, minh):
                acc = 0
                for fr in range(int(minh), int(maxh)):
                    if fr in curr_person_boxes[:,0] and fr in curr_hand_boxes[:,0]:
                        tl_p = curr_person_boxes[curr_person_boxes[:,0]==fr, 2:4]
                        br_p = curr_person_boxes[curr_person_boxes[:,0]==fr, 2:4] + curr_person_boxes[curr_person_boxes[:,0]==fr, 4:6]
                        tl_h = curr_hand_boxes[curr_hand_boxes[:,0]==fr, 2:4]
                        br_h = curr_hand_boxes[curr_hand_boxes[:,0]==fr, 2:4] + curr_hand_boxes[curr_hand_boxes[:,0]==fr, 4:6]
                        tl = np.asarray([np.maximum(tl_h[0][0],tl_p[0][0]),np.maximum(tl_h[0][1],tl_p[0][1])])
                        br = np.asarray([np.minimum(br_h[0][0],br_p[0][0]),np.minimum(br_h[0][1],br_p[0][1])]) 
                        wh = np.maximum(0., br - tl)
                        area_intersection = wh.prod()
                        acc += (area_intersection > 1).sum()
                        person_mat[int(h), int(p)] = acc
        for b in bids:
            curr_bin_boxes = bin_boxes[bin_boxes[:,1]==b]
            minb = min(curr_bin_boxes[:,0])
            maxb = max(curr_bin_boxes[:,0])
            if min(maxb, maxh) > max(minb, minh):
                acc = 0
                for fr in range(int(minh), int(maxh)):
                    if fr in curr_bin_boxes[:,0] and fr in curr_hand_boxes[:,0]:
                        tl_b = curr_bin_boxes[curr_bin_boxes[:,0]==fr, 2:4]
                        br_b = curr_bin_boxes[curr_bin_boxes[:,0]==fr, 2:4] + curr_bin_boxes[curr_bin_boxes[:,0]==fr, 4:6]
                        tl_h = curr_hand_boxes[curr_hand_boxes[:,0]==fr, 2:4]
                        br_h = curr_hand_boxes[curr_hand_boxes[:,0]==fr, 2:4] + curr_hand_boxes[curr_hand_boxes[:,0]==fr, 4:6]
                        tl = np.asarray([np.maximum(tl_h[0][0],tl_b[0][0]),np.maximum(tl_h[0][1],tl_b[0][1])])
                        br = np.asarray([np.minimum(br_h[0][0],br_b[0][0]),np.minimum(br_h[0][1],br_b[0][1])])
                        wh = np.maximum(0., br - tl)
                        area_intersection = wh.prod()
                        acc += (area_intersection > 1).sum()
                        bin_mat[int(h), int(b)] = acc
    pid = np.argmax(person_mat, axis=1)
    bid = np.argmax(bin_mat, axis=1)
    sim_mat = {}
    pairs = []  
    for i,b in enumerate(bid):
        if bin_mat[i,b] > 1 and person_mat[i,pid[i]] > 1:
        # if bin_mat[i,b] > 5 and person_mat[i,pid[i]] > 5:
            pairs.append((b,pid[i]))
    pairs = Counter(pairs)
    for key in pairs.keys():
        if key[0] not in sim_mat.keys():
            sim_mat[key[0]] = key[1]
        else:
            if pairs[key] > pairs[(key[0],sim_mat[key[0]])]:
                sim_mat[key[0]] = key[1]
            elif pairs[key] < pairs[(key[0],sim_mat[key[0]])]:
                continue
            else:
                indices = [i for i, x in enumerate(bid) if x == key[0]]
                hand_id = indices[np.argmax(bin_mat[indices, key[0]])]
                sim_mat[key[0]] = pid[hand_id]
    # sim_mat = {}
    # for b in bids:
    #     b = int(b)
    #     mask = bin_mat[:,b] != 0
    #     person_hands = np.zeros(len(pids))
    #     for p in pids:
    #         p = int(p)
    #         person_hands[p] = np.sum(person_mat[mask,p])
    #     match_person = np.argmax(person_hands)
    #     if person_hands[match_person] > 1:
    #         sim_mat[b] = match_person
    return sim_mat

def location_matching(person_boxes, bin_boxes):
    pids = np.unique(person_boxes[:,1])
    bids = np.unique(bin_boxes[:,1])
    dist_mat = np.ones((len(pids), int(max(bids)+1))) * 1e8
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
                for fr in range(int(max(minp, minb))+50, int(min(maxp, maxb))):
                    xy_p = curr_person_boxes[curr_person_boxes[:,0]==fr, 2:4] + curr_person_boxes[curr_person_boxes[:,0]==fr, 4:6]
                    xy_b = curr_bin_boxes[curr_bin_boxes[:,0]==fr, 2:4] + curr_bin_boxes[curr_bin_boxes[:,0]==fr, 4:6]
                    if xy_p.size > 0 and xy_b.size > 0:
                        acc_d += np.linalg.norm(0.5*(xy_p - xy_b))
                        acc += 1
                if acc > 0:
                    avg_d = acc_d / acc
                    dist_mat[int(p), int(b)] = avg_d
    return dist_mat    

def associate(exp, startf=0, endf=100000, vis=True, fps=20.0):
    video_file = "demo/" + exp + ".mp4"
    person_track_file = 'demo/' + exp + '_person.txt'
    bin_track_file = 'demo/' + exp + '_bin.txt'
    hand_track_file = 'demo/' + exp + '_hand.txt'

    person_boxes = np.loadtxt(person_track_file, delimiter=',')
    bin_boxes = np.loadtxt(bin_track_file, delimiter=',')
    hand_boxes = np.loadtxt(hand_track_file, delimiter=',')
        

    location_dist = location_matching(person_boxes, bin_boxes) 
    bin_ass = np.argmin(location_dist, axis=0)
    sim_mat = hand_matching(hand_boxes, person_boxes, bin_boxes)

    for b in sim_mat.keys():
        bin_ass[b] = sim_mat[b]
        
    out_text = './demo/' + exp + '_association_person.txt'
    f = open(out_text, 'w')
    tmp = [str(x) for x in bin_ass]
    print(', '.join(tmp),file=f)
    f.close()

    capture = cv2.VideoCapture(video_file)
    capture.set(1, startf)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    i = startf - 1
    #i = 55
    
    if vis:
        out = cv2.VideoWriter('./demo/' + exp + '_SORT_Tracking.mp4', fourcc, fps, (1920, 1080))
        while True:
            flag, frame = capture.read()
            if frame is not None:
                frame = frame[:,:,::-1]
                image = Image.fromarray(frame)
                i += 1
            else: break
            if i > endf: break

            curr_person_boxes = person_boxes[person_boxes[:,0] == i]
            curr_bin_boxes    = bin_boxes[bin_boxes[:,0] == i]
            curr_hand_boxes    = hand_boxes[hand_boxes[:,0] == i]

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
                
            image_np = write_tracks_on_image_array(
                image,
                curr_hand_boxes[:,2:],
                curr_hand_boxes[:,1],
                bin_ass,
                'hand')

            image_np = image_np[:,:,::-1]
            out.write(image_np)
            print('%d frames processed!' % (i - startf + 1))
