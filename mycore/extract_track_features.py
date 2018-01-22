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
PATH_TO_CKPT = 'model/alert/frozen_inference_graph.pb'
PATH_TO_CKPT_CLS = 'model/imagenet/classify_image_graph_def.pb'
NUM_CLASSES = 2

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def write_tracks_on_image_array(image_pil, boxes, ids, clss):
    for box, i in zip(boxes, ids):
        write_tracks_on_image(image_pil, box, i, clss)
    image = np.array(image_pil)
    return image

def write_tracks_on_image(image, box, id, clss):
    txtcolor = 'red'
    if clss == 'person':
        boxcolor = 'blue'
    else:
        boxcolor = 'green'    
    draw = ImageDraw.Draw(image)

    (left, right, top, bottom) = (box[0], box[0]+box[2], box[1], box[1]+box[3])
    try:
        font = ImageFont.truetype('demo/arial.ttf', 30)
    except IOError:
        font = ImageFont.load_default()

    display_str = clss + ' ' + str(int(id))
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

def get_boxes_from_image(image, boxes):
    image_data = []
    h, w, _ = image.shape
    for box in boxes:
        x1 = max(int(box[0]), 0)
        y1 = max(int(box[1]), 0)
        x2 = min(int(x1 + box[2]), w)
        y2 = min(int(y1 + box[3]), h)
        image_data.append(resize(image[y1:y2, x1:x2, :], (300, 300), preserve_range=True).astype(np.uint8))
    return image_data


def extract_features(exp, startf=0, endf=100000, vis=True, fps=20.0):
    video_file = "demo/" + exp + ".mp4"
    person_track_file = 'demo/' + exp + '_person.txt'
    bin_track_file = 'demo/' + exp + '_bin.txt'

    capture = cv2.VideoCapture(video_file)
    capture.set(1, startf)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./demo/' + exp + '_SORT_Tracking.avi', fourcc, fps, (1920, 1080))
    
    person_features_npy = './demo/' + exp + '_features_person.npy'
    bin_features_npy = './demo/' + exp + '_features_bin.npy'

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
            np.swapaxes(frame, 0, 2)
            image = Image.fromarray(frame)
            i += 1
        else: break
        if i > endf: break
        
        curr_person_boxes = person_boxes[person_boxes[:,0] == i]
        curr_bin_boxes    = bin_boxes[bin_boxes[:,0] == i]

        if curr_person_boxes.size > 0:
            with classification_graph.as_default():
                image_data = get_boxes_from_image(np.squeeze(image), curr_person_boxes[:,2:])
                feature = classification_graph.get_tensor_by_name('pool_3:0')

                for sub_image, box in zip(image_data, curr_person_boxes[:,:2]):
                    curr_person_feature = sess_cls.run(feature,
                                               feed_dict={'DecodeJpeg:0': sub_image})
                    curr_person_feature = np.squeeze(curr_person_feature)
                    person_features.append(np.concatenate([box.astype(np.float32), curr_person_feature]))

        if curr_bin_boxes.size > 0:
            with classification_graph.as_default():
                image_data = get_boxes_from_image(np.squeeze(image), curr_bin_boxes[:,2:])
                feature = classification_graph.get_tensor_by_name('pool_3:0')

                for sub_image, box in zip(image_data, curr_bin_boxes[:,:2]):
                    curr_bin_feature = sess_cls.run(feature,
                                               feed_dict={'DecodeJpeg:0': sub_image})
                    curr_bin_feature = np.squeeze(curr_bin_feature)
                    bin_features.append(np.concatenate([box.astype(np.float32), curr_bin_feature]))
            

        if vis:
            image_np = write_tracks_on_image_array(
                image,
                curr_person_boxes[:,2:],
                curr_person_boxes[:,1],
                'person')

            image_np = write_tracks_on_image_array(
                image,
                curr_bin_boxes[:,2:],
                curr_bin_boxes[:,1],
                'bin')

            np.swapaxes(image_np, 0, 2)
            out.write(image_np)
        print('%d frames processed!' % (i - startf + 1))


    np.save(person_features_npy, np.array(person_features))
    np.save(bin_features_npy, np.array(bin_features))   
