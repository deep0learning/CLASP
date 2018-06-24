import os
import shutil

anno_root = '/home/hxw/data/alert/CLASP_labeling/CLASP_labels/'

anno_names = ['5A_Take1_C9', '5A_Take1_C11', '7A_C9', '7A_C11', '9A_C9',
               '9A_C11', '10A_C9', '10A_C11']

save_dir = '/home/hxw/projects/alert/deep_detection/data/alert/annotations/'

for i in range(len(anno_names)):
    anno_dir = anno_root + anno_names[i] + '/'
    file_names = os.listdir(anno_dir)
    for file in file_names:
        frame_id = file[5:9]
        src_file = anno_dir + file
        dst_file = save_dir + anno_names[i] + '_' + frame_id + '.jpg.txt'
        shutil.copyfile(src_file, dst_file)
