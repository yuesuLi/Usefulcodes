import numpy as np
import os
import cv2

w = 1430
h = 780
size = (w,h)
fps = 2
file_path = '/media/personal_data/zhangq/DeepSORT/yolov5-master/results/'
save_path = '/home/zhangq/Desktop/zhangq/UsefulCode/' + 'test.mp4'
videowrite = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))


# dir = "/home/zhangq/Desktop/zhangq/UsefulCode/imgs/"
# frame_cnt = 0
# for root, dirs, files in os.walk(dir):
#     for file in files:
#         if file[-3:] == 'png':
#             frame_cnt += 1
#             source = str(os.path.join(root, file))
#             videowrite.write()
            # print(int(file.split('.')[0][-5:])*1000 + int(file.split('.')[1][:]))

dir = '/home/zhangq/Desktop/zhangq/UsefulCode/0'
iter_my = os.listdir(dir)
iter_my.sort(key=lambda x:(int(x.split('.')[0][5:])))

for spath in iter_my:
    # print(os.path.join(dir, spath))
    img_name = str(os.path.join(dir, spath))
    img = cv2.imread(img_name)
    videowrite.write(img)


# base_path = '/home/zhangq/Desktop/ourDataset/v1.0_label/20211027_2_group0079_109frames_22labeled'
# folders = os.listdir(base_path)
# folders = sorted(folders)
# img_names = []
# for folder in folders:
#     camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
#     for file in os.listdir(camera_path):
#         if file[-3:] == 'png':
#             img_path = os.path.join(camera_path, file)
#             # img_names.append(img_path)
#             img = cv2.imread(img_path)
#             videowrite.write(img)
# #
# # print('img_names', img_names)



videowrite.release()
cv2.destroyWindow()

