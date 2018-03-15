# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

import sys
sys.path.insert(0, "/home/hxw/projects/alert/deep_detection/mylib/deep_sort")


from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

import pdb


def gather_sequence_info(sequence_dir, detection_file, capture):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None

    flag, frame = capture.read()    

    if frame is not None:
        image_size = frame.shape[0:2]
    else:
        image_size = None


    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())

    update_ms = None

    feature_dim = detections.shape[1] - 7 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir.split('.')[0]),
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


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


def create_detections(detection_mat, frame_idx, image_size, clss, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    if clss == 'person':
        interested_class = 1
    elif clss == 'bin':
        interested_class = 2
    else:
        raise Exception('Unknown class id!')        

    detection_list = []
    for row in detection_mat[mask]:
        tclss, rbbox, confidence, feature = row[1], row[2:6], row[6], row[7:]
        bbox = box_transform(rbbox, image_size)
        if bbox[3] < min_height:
            continue
        if tclss == interested_class:     
            detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, clss):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    capture = cv2.VideoCapture(sequence_dir)
    seq_info = gather_sequence_info(sequence_dir, detection_file, capture)
    capture.set(1, seq_info['min_frame_idx'])
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, seq_info['image_size'], clss, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            #capture.set(1, frame_idx)
            _, image = capture.read()
            #pdb.set_trace()
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Post-process results.
    results_array = np.array(results)
    track_idxs = results_array[:, 1]
    unique, counts = np.unique(track_idxs, return_counts=True)
    #pdb.set_trace()
    for u, c in zip(unique, counts):
        if c < 200:
            results_array = results_array[track_idxs!=u]
            track_idxs = track_idxs[track_idxs!=u]

    foo, unique_idxs = np.unique(track_idxs, return_inverse=True)
    results_array[:,1] = unique_idxs
    results_processed = results_array.tolist()

    # Store results.
    f = open(output_file+'_'+clss+'.txt', 'w')
    for row in results_processed:
        print('%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.1)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=boolean_string)
    parser.add_argument(
        "--clss", help="Object class to tracking (person/bin)",
        default='person', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display, args.clss)

def tracking(exp, clss):
    sequence_dir = './demo/%s.mp4' % exp
    detection_file = './demo/%s_FRCNN_DET.npy' % exp
    output_file = './demo/%s' % exp
    min_confidence = 0.8
    nms_max_overlap = 1.0
    min_detection_height = 0
    max_cosine_distance = 0.2
    nn_budget=30
    display=False
    run(
        sequence_dir, detection_file, output_file,
        min_confidence, nms_max_overlap, min_detection_height,
        max_cosine_distance, nn_budget, display, clss)    
