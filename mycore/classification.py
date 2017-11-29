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
from utils import label_map_util
from utils import visualization_utils as vis_util
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from itertools import compress


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'model/alert/frozen_inference_graph.pb'
PATH_TO_CKPT_CLS = 'model/imagenet/classify_image_graph_def.pb'
PATH_TO_LABELS = 'data/alert/alert_label_map.pbtxt'
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

intrst_id = [220, 222, 375, 529, 535, 551, 574, 748, 750,
             751, 757, 760, 766, 774, 777, 788, 794, 817, 825,
             837, 845, 847, 867, 877, 879, 881, 901, 909,
             910, 918, 928, 930, 934, 937, 939, 958, 961, 964,
             965, 966, 987]


def write_classes_on_image_array(image,
                                 boxes,
                                 classes):
    image_pil = Image.fromarray(np.uint8(image))
    for box, clss in zip(boxes, classes):
        write_classes_on_image(image_pil, box, clss)
    np.copyto(image, np.array(image_pil))
    return image


def write_classes_on_image(image, box, classes):
    color = 'red'
    draw = ImageDraw.Draw(image)
    w, h = image.size
    xmin = box[1]
    ymin = box[0]
    xmax = box[3]
    ymax = box[2]

    (left, right, top, bottom) = (xmin * w, xmax * w,
                                  ymin * h, ymax * h)
    try:
        font = ImageFont.truetype('demo/arial.ttf', 30)
    except IOError:
        font = ImageFont.load_default()

    text_bottom = bottom

    for display_str in classes[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.5 * text_height)
        # draw.rectangle(
        #     [(left, text_bottom - text_height - 2 * margin), (left + text_width,
        #                                                       text_bottom)],
        #     fill='red')
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='blue',
            font=font)
        text_bottom = text_bottom - text_height - 2 * margin


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = 'model/imagenet/imagenet_2012_challenge_label_map_proto.pbtxt'
        if not uid_lookup_path:
            uid_lookup_path = 'model/imagenet/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def load_images_into_numpy_array(images):
    (im_width, im_height) = images[0].size
    images_np = np.zeros((len(images), im_height, im_width, 3), dtype=np.uint8)
    for i in range(len(images)):
        images_np[i] = np.array(images[i].getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    return images_np, len(images)


def get_boxes_from_image(image, boxes):
    image_data = []
    h, w, _ = image.shape
    for box in boxes:
        x1 = int(w * box[1])
        y1 = int(h * box[0])
        x2 = int(min(w * box[3], w))
        y2 = int(min(h * box[2], h))
        image_data.append(resize(image[y1:y2, x1:x2, :], (300, 300), preserve_range=True).astype(np.uint8))
    return image_data


def IOU_filter(box, bboxes, iitems):
    ostrs = []
    if len(bboxes) > 0:
        w = 1920
        h = 1080
        max_iou = 0.0
        for i in range(len(bboxes)):
            for j in range(bboxes[i].shape[0]):
                box2 = bboxes[i][j]
                min_x = min(box[1], box2[1]) * w
                min_y = min(box[0], box2[0]) * h
                max_x = max(box[3], box2[3]) * w
                max_y = max(box[2], box2[2]) * h
                w1 = (box[3] - box[1]) * w
                h1 = (box[2] - box[0]) * h
                w2 = (box2[3] - box2[1]) * w
                h2 = (box2[2] - box2[0]) * h

                uw = max_x - min_x
                uh = max_y - min_y

                # intersection width and height
                cw = w1 + w2 - uw
                ch = h1 + h2 - uh

                # not overlapped
                if cw <= 0 or ch <= 0:
                    iou = 0.0

                else:
                    area1 = w1 * h1
                    area2 = w2 * h2
                    carea = cw * ch  # intersection area
                    uarea = area1 + area2 - carea
                    iou = carea / uarea

                strs = iitems[i][j]
                if iou > max_iou:
                    ostrs = strs
                    max_iou = iou

        if max_iou < 0.5:
            ostrs = []

    return ostrs


def classification(exp, startf=0, endf=100000, vis=True, fps=20.0):
    video_file = "demo/" + exp + ".mp4"
    capture = cv2.VideoCapture(video_file)
    capture.set(1, startf)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./demo/' + exp + '_FRCNN_CLS.avi', fourcc, fps, (1920, 1080))
    otxt = open('./demo/' + exp + '_FRCNN_CLS.txt', 'w')

    itxt = open('./demo/' + exp + '_FRCNN_DET.txt', 'r')

    classification_graph = tf.Graph()
    with classification_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT_CLS, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    sess_cls = tf.Session(graph=classification_graph)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    i = startf - 1
    oboxes = []
    oitems = []
    while True:
        flag, frame = capture.read()
        if frame is not None:
            image = Image.fromarray(frame)
            i += 1
        else: break
        if i > endf: break

        boxes, scores, classes = [], [], []
        while True:
            fsd = itxt.tell()
            line = itxt.readline().split(',')

            if line == ['']: break
            if int(line[0]) == i:
                classes.append(int(line[1]))
                scores.append(float(line[6].split('\n')[0]))
                boxes.append([float(line[2]), float(line[3]), float(line[4]), float(line[5])])
            if int(line[0]) > i:
                itxt.seek(fsd)
                break

        if line == ['']: break

        batch_image_np, bsize = load_images_into_numpy_array([image])

        # Load boxes
        (boxes, scores, classes) = np.array(boxes), np.array(scores), np.array(classes)

        if vis:
            image_np = vis_util.visualize_boxes_and_labels_on_image_array(
                np.squeeze(batch_image_np),
                boxes,
                classes.astype(np.int32),
                scores,
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.1,
                line_thickness=8)

            if boxes.size > 0:
                with classification_graph.as_default():
                    image_data = get_boxes_from_image(np.squeeze(image), boxes)
                    logits = classification_graph.get_tensor_by_name('softmax/logits:0')

                    all_items = []
                    for sub_image, box in zip(image_data, boxes):
                        ostrs = IOU_filter(box, oboxes, oitems)
                        if len(ostrs) > 0:
                            all_items.append(ostrs)
                        else:
                            items = []
                            predictions = sess_cls.run(logits,
                                                       feed_dict={'DecodeJpeg:0': sub_image})
                            predictions = np.squeeze(predictions)
                            predictions = predictions[intrst_id]
                            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

                            node_lookup = NodeLookup()
                            top_k = predictions.argsort()[-3:][::-1]
                            for node_id in top_k:
                                human_string = node_lookup.id_to_string(intrst_id[node_id])
                                score = predictions[node_id]
                                if score > 0.1:
                                    items.append(human_string.split(',')[0])
                                print('%s (score = %.5f)' % (human_string, score))
                            all_items.append(items)

                    image_np = write_classes_on_image_array(
                        np.squeeze(image_np),
                        boxes[classes == 2.0],
                        list(compress(all_items, classes == 2.0)))

                    oitems.append(all_items)
                    oboxes.append(np.copy(boxes))

                    if len(oboxes) > 10:
                        oitems.pop(0)
                        oboxes.pop(0)

                    if i % 60 == 0:
                        oitems[:] = []
                        oboxes[:] = []

            out.write(image_np)
        print('%d frames processed!' % (i - startf + 1))

    itxt.close()
    otxt.close()
