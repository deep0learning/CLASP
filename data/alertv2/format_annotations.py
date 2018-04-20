import os
import shutil
import pdb

# anno_root = '/home/hxw/data/alert/review_data/annotationsColored/'

# anno_names = ['2_C11', '3_C9']

# save_dir = '/home/hxw/projects/alert/deep_detection/data/alertv2/annotations/'

# for i in range(len(anno_names)):
#     anno_dir = anno_root + anno_names[i] + '/labels/'
#     file_names = os.listdir(anno_dir)
#     for file in file_names:
#       file_name = file.split('.')[0]
#         frame_id = file_name[5:]
#         src_file = anno_dir + file
#         dst_file = save_dir + anno_names[i] + '_' + '%04d' % int(frame_id) + '.jpg.txt'
#         print dst_file
#         shutil.copyfile(src_file, dst_file)

anno_file = '/home/hxw/data/alert/review_data/master-sikka-exp5a-logfile.txt'

save_dir = '/home/hxw/projects/alert/deep_detection/data/alertv2/annotations/'

with open(anno_file) as f:
    content = f.readlines()

content = [x.strip() for x in content]
pdb.set_trace()

for line in content:
    if line.startswith('# frame'):
        frame_id = int(line.split(" ")[2])
    elif line.startswith('loc'):
        # find camera id, class, locations.
        clss = line[line.find("type")+6:].split(" ")[0]
        cam = line[line.find("camera-num")+12:].split(" ")[0]
        bbox = line[line.find("bb")+4:].split(" ")[0:4]
        
        dst_file = save_dir + '5A_' + 'C' + cam + '_' '%04d' % (frame_id) + '.jpg.txt'

        newline = "%s %d %d %d %d\r\n" % (clss, int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1]))

        with open(dst_file, "a") as newfile:
            newfile.write(newline)
        print dst_file    
            

