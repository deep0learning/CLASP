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


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT_CLS = 'model/market/frozen_graph.pb'
NUM_CLASSES = 2

def get_boxes_from_image(image, boxes):
    image_data = []
    h, w, _ = image.shape
    for box in boxes:
        x1 = max(int(box[0]), 0)
        y1 = max(int(box[1]), 0)
        x2 = min(int(x1 + box[2]), w)
        y2 = min(int(y1 + box[3]), h)
        image_data.append(((resize(image[y1:y2, x1:x2, :], (299, 299), preserve_range=True).astype(np.float32))/255-0.5)/2)
    return image_data


def extract_features(exp, startf=0, endf=100000, vis=False, fps=40.0):
    video_file = "result/original/" + exp + ".mp4"
    person_track_file = 'result/tracking/' + exp + '_person.txt'
    bin_track_file = 'result/tracking/' + exp + '_bin.txt'

    capture = cv2.VideoCapture(video_file)
    capture.set(1, startf)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('./result/tracking/' + exp + '_SORT_Tracking.avi', fourcc, fps, (1920, 1080))
    
    person_features_npy = './result/tracking/' + exp + '_features_person.npy'
    bin_features_npy = './result/tracking/' + exp + '_features_bin.npy'

    person_boxes = np.loadtxt(person_track_file, delimiter=',')
    bin_boxes = np.loadtxt(bin_track_file, delimiter=',')
    

    classification_graph = tf.Graph()
    with classification_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_CLS, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    sess_cls = tf.Session(graph=classification_graph)

    i = startf - 1
    person_features = []
    bin_features = []

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

        if curr_person_boxes.size > 0:
            with classification_graph.as_default():
                image_data = get_boxes_from_image(np.squeeze(image), curr_person_boxes[:,2:])

                feature = classification_graph.get_tensor_by_name('InceptionV3/Logits/Dropout_1b/Identity:0')

                for sub_image, box in zip(image_data, curr_person_boxes[:,:2]):
                    sub_image = sub_image[np.newaxis, :]
                    curr_person_feature = sess_cls.run(feature,
                                               feed_dict={'Placeholder:0': sub_image})
                    curr_person_feature = np.squeeze(curr_person_feature)
                    person_features.append(np.concatenate([box.astype(np.float32), curr_person_feature]))

        if curr_bin_boxes.size > 0:
            with classification_graph.as_default():
                image_data = get_boxes_from_image(np.squeeze(image), curr_bin_boxes[:,2:])
                feature = classification_graph.get_tensor_by_name('InceptionV3/Logits/Dropout_1b/Identity:0')

                for sub_image, box in zip(image_data, curr_bin_boxes[:,:2]):
                    sub_image = sub_image[np.newaxis, :]
                    curr_bin_feature = sess_cls.run(feature,
                                               feed_dict={'Placeholder:0': sub_image})
                    curr_bin_feature = np.squeeze(curr_bin_feature)
                    bin_features.append(np.concatenate([box.astype(np.float32), curr_bin_feature]))
            
        print('%d frames processed!' % (i - startf + 1))


    np.save(person_features_npy, np.array(person_features))
    np.save(bin_features_npy, np.array(bin_features))   
