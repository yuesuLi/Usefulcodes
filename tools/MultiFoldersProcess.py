import os
import re
import sys
import time
import math
import json
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.patches as mpatches


base_path = '/media/ourDataset/v1.0_label/20211027_1_group0028_134frames_27labeled'
# base_path = './20211027_2_group0051_72frames_15labeled'
save_path = './save/20211027_1_group0026/'

folders = os.listdir(base_path)
folders = sorted(folders)
num_frame = -1
for folder in folders:
    num_frame += 1
    print('num_frame:', num_frame)
    camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
    for file in os.listdir(camera_path):
        if file[-3:] == 'png':
            img_path = os.path.join(camera_path, file)
    lidar_path = os.path.join(base_path, folder, 'VelodyneLidar')
    for file in os.listdir(lidar_path):
        if file[-3:] == 'pcd':
            Lidar_pcd_path = os.path.join(lidar_path, file)
        if file[-4:] == 'json':
            Lidar_json_path = os.path.join(lidar_path, file)
    radar_path = os.path.join(base_path, folder, 'OCULiiRadar')
    for file in os.listdir(radar_path):
        if file[-3:] == 'pcd':
            OCU_pcd_path = os.path.join(radar_path, file)
        if file[-4:] == 'json':
            OCU_radar_json_path = os.path.join(radar_path, file)
    ti_path = os.path.join(base_path, folder, 'TIRadar')
    for file in os.listdir(ti_path):
        if file[-3:] == 'pcd':
            TI_pcd_path = os.path.join(ti_path, file)
        if file[-4:] == 'json':
            TI_radar_json_path = os.path.join(ti_path, file)

    img = cv.imread(img_path)
    width, height = img.shape[1], img.shape[0]


