import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont



def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin

anno_dir = '/home/hxw/projects/alert/deep_detection/data/alert/annotations/'
image_dir = '/home/hxw/projects/alert/deep_detection/data/alert/images/'

image_files = os.listdir(image_dir)

fig, ax = plt.subplots(1)
plt.ion()

error_file = open('anno_errors.txt', 'w')

for i in range(len(image_files)):
    im = Image.open(image_dir+image_files[i])
    anno_file = anno_dir + image_files[i] + '.txt'

    with open(anno_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    bbox = []

    if len(content) > 1:
        for j in range(len(content)-1):
            foo = content[j+1].split()
            det = map(int, foo[1:5])
            bbox.append(det)

    if bbox:
        for b in bbox:
            draw_bounding_box_on_image(im, b[1], b[0], b[1]+b[3], b[0]+b[2], use_normalized_coordinates=False)

    ax.imshow(im)
    plt.show()
    print('checking image file %s' % image_files[i])
    feedback = raw_input()
    if feedback == ' ':
        print('Not accurate annotation.')
        error_file.write(image_files[i] + '\r\n')
