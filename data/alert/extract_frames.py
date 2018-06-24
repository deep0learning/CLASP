import numpy as np
import cv2


def extract_frames(video_dir, image_dir, save_name, rate):
    print('Processing Video %s' % video_dir)
    capture = cv2.VideoCapture(video_dir)
    while not capture.isOpened():
        capture = cv2.VideoCapture(video_file)
        cv2.waitKey(1000)
        print("Wait for the header")

    i = 0
    while True:
        flag, frame = capture.read()

        if frame is not None:
            print("Processed %d frames" % i)

            if i % rate == 0:
                cv2.imwrite(image_dir+save_name+'_%04d.jpg' % i, frame)
            i += 1
        else:
            break
            # save_pickle(fg, './result/Camera_8_%s_post.pkl' % type(bgs).__name__)

    cv2.destroyAllWindows()

video_root = '/home/hxw/data/alert/alert-data-new/'

video_files = ['07212017_EXPERIMENT_5A/TAKE_1/mp4s/Camera_9.mp4',
               '07212017_EXPERIMENT_5A/TAKE_1/mp4s/Camera_11.mp4',
               '07212017_EXPERIMENT_7A/mp4s/Camera_9.mp4',
               '07212017_EXPERIMENT_7A/mp4s/Camera_11.mp4',
               '07212017_EXPERIMENT_9A/mp4s/Camera_9.mp4',
               '07212017_EXPERIMENT_9A/mp4s/Camera_11.mp4',
               '07212017_EXPERIMENT_10A/mp4s/Camera_9.mp4',
               '07212017_EXPERIMENT_10A/mp4s/Camera_11.mp4']

video_names = ['5A_Take1_C9', '5A_Take1_C11', '7A_C9', '7A_C11', '9A_C9',
               '9A_C11', '10A_C9', '10A_C11']

image_dir = '/home/hxw/projects/alert/deep_detection/data/alert/images/'

for i in range(len(video_names)):
    video_dir = video_root + video_files[i]
    save_name = video_names[i]
    extract_frames(video_dir, image_dir, save_name, 30)

