from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cv2

import numpy as np
import PIL.Image as Image

import sys
sys.path.insert(0, "/project/google-surv/alert/models/research/object_detection")
from utils import label_map_util
from utils import visualization_utils as vis_util

import pdb

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'model/alert/frozen_inference_graph.pb'
PATH_TO_CKPT_CLS = 'model/imagenet/classify_image_graph_def.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/alert/alert_label_map.pbtxt'
NUM_CLASSES = 2

def load_images_into_numpy_array(images):
    (im_width, im_height) = images[0].size
    images_np = np.zeros((len(images), im_height, im_width, 3), dtype=np.uint8)
    for i in range(len(images)):
        images_np[i] = np.array(images[i].getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

    return images_np, len(images)


def detection(exp, dtype='perbin', startf=0, endf=100000, THR=0.85, vis=True, fps=20.0):
    video_file = "demo/" + exp + ".mp4"
    capture = cv2.VideoCapture(video_file)
    capture.set(1, startf)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    onpy = './demo/' + exp + '_FRCNN_DET.npy'

    if dtype == 'hand':
        PATH_TO_CKPT = 'model/alert/hand/frozen_inference_graph.pb'
        #PATH_TO_CKPT = '/project/google-surv/alert/models/research/object_detection/hands_log/saved_graph/output_inference_graph/frozen_inference_graph.pb'
        PATH_TO_LABELS = 'data/alert/hands_label_map.pbtxt'
        onpy = './demo/' + exp + '_FRCNN_DET__hand.npy'
        NUM_CLASSES = 1
    else:
        PATH_TO_CKPT = 'model/alert/frozen_inference_graph.pb'
        PATH_TO_CKPT_CLS = 'model/imagenet/classify_image_graph_def.pb'
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = 'data/alert/alert_label_map.pbtxt'
        NUM_CLASSES = 2

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    sess_det = tf.Session(graph=detection_graph)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_results = []

    i = startf-1
    while True:
        flag, frame = capture.read()
        if frame is not None:
            frame = frame[:,:,::-1]
            image = Image.fromarray(frame)
            i += 1
        else:
            break
        if i > endf:
            break
        with detection_graph.as_default():
            if dtype == 'hand':
                box_features = detection_graph.get_tensor_by_name('SecondStageFeatureExtractor/InceptionResnetV2/Conv2d_7b_1x1/Relu:0')
            else:
                box_features = detection_graph.get_tensor_by_name('SecondStageFeatureExtractor/resnet_v1_101/block4/unit_3/bottleneck_v1/Relu:0')

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            # run one forward pass on this batch
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            batch_image_np, bsize = load_images_into_numpy_array([image])

            # Actual detection.
            (boxes, scores, classes, num, features) = sess_det.run(
              [detection_boxes, detection_scores, detection_classes, num_detections, box_features],
              feed_dict={image_tensor: batch_image_np})
              
            if len(scores.shape)<2:
                scores = np.expand_dims(scores, axis=0)
            
            features = np.expand_dims(np.mean(features, axis=(1,2)), axis=0)

            if vis:
                out = cv2.VideoWriter('./demo/' + exp + '_FRCNN_DET.mp4', fourcc, fps, (1920, 1080))
                image_np = vis_util.visualize_boxes_and_labels_on_image_array(
                  np.squeeze(batch_image_np),
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  min_score_thresh=THR,
                  line_thickness=8)
                image_np = image_np[:,:,::-1]
                out.write(image_np)
            
            boxes, scores, classes, features = boxes[scores > THR], scores[scores > THR], classes[scores > THR], features[scores > THR]
            
            for k in range(len(scores)):
                if dtype == 'hand':
                    current_result = np.zeros(shape=1543)
                else:
                    current_result = np.zeros(shape=(2055))
                current_result[0] = i
                current_result[1] = classes[k]
                current_result[2:6] = boxes[k][0:4]
                current_result[6] = scores[k]
                current_result[7:] = features[k]
                detection_results.append(current_result)
                #otxt.write('%d,%d,%.3f,%.3f,%.3f,%.3f,%.2f, %s\n' % (i, classes[k], boxes[k][0], boxes[k][1],
                #                                          boxes[k][2], boxes[k][3], scores[k], ",".join(['%f' % w for w in features[k]])))

            print('%d frames processed!' % (i-startf+1))

    np.save(onpy, np.array(detection_results))
