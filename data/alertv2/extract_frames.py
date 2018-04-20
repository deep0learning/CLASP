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

video_root = '/home/hxw/data/alert/review_data/'

video_files = ['cam2exp5a.mp4',
               'cam5exp5a.mp4',
               'cam9exp3.mp4',
               'cam9exp5a.mp4',
               'cam11exp2.mp4',
               'cam11exp5a.mp4',
               'cam13exp5a.mp4']

video_names = ['5A_C2', '5A_C5', '3_C9', '5A_C9', '2_C11',
               '5A_C11', '5A_C13']

image_dir = '/home/hxw/projects/alert/deep_detection/data/alertv2/images/'

for i in range(len(video_names)):
    video_dir = video_root + video_files[i]
    save_name = video_names[i]
    extract_frames(video_dir, image_dir, save_name, 1)

