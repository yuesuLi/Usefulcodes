import argparse
import os
import sys
from pathlib import Path
import numpy as np
import json

import re
import mayavi
import matplotlib.pyplot as plt

import cv2
import torch
numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)

# cmap = plt.cm.get_cmap('hsv', 256)
# cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

def parse_header(lines):
    '''Parse header of PCD files'''
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn(f'warning: cannot understand line: {ln}')
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = map(int, value.split())
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()

    if 'count' not in metadata:
        metadata['count'] = [1] * len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata

def _build_dtype(metadata):
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type] * c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_ascii_pc_data(f, dtype, metadata):
    # for radar point
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    # for lidar point
    rowstep = metadata['points'] * dtype.itemsize
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    raise NotImplemented

def read_pcd(pcd_path, pts_view=False):
    # pcd = o3d.io.read_point_cloud(pcd_path)
    f = open(pcd_path, 'rb')
    header = []
    while True:
        ln = f.readline().strip()  # ln is bytes
        ln = str(ln, encoding='utf-8')
        # ln = str(ln)
        header.append(ln)
        # print(type(ln), ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or "binary_compressed"')

    points = np.concatenate([pc_data[metadata['fields'][0]][:, None],
                             pc_data[metadata['fields'][1]][:, None],
                             pc_data[metadata['fields'][2]][:, None]], axis=-1)
    # print(points.shape)
    return points

def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        return json_data

# base_path = '/home/zhangq/Desktop/ourDataset/v1.0_label/20211025_2_group0013_351frames_71labeled'
# base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/tmp2'
# base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train'
base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled/20211025_1_group0012_frame0050_labeled'

folders = os.listdir(base_path)
folders = sorted(folders)
frame_num = -1
for folder in folders:
    camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
    for file in os.listdir(camera_path):
        if file[-3:] == 'png':
            img_path = os.path.join(camera_path, file)
        if file[-4:] == 'json':
            img_json = os.path.join(camera_path, file)
    # lidar_path = os.path.join(base_path, folder, 'VelodyneLidar')
    # for file in os.listdir(lidar_path):
    #     if file[-3:] == 'pcd':
    #         pcd_lidar = os.path.join(lidar_path, file)
    #     if file[-4:] == 'json':
    #         calib_lidar = os.path.join(lidar_path, file)
    ti_path = os.path.join(base_path, folder, 'TIRadar')
    for file in os.listdir(ti_path):
        if file[-3:] == 'pcd':
            TI_pcd_path = os.path.join(ti_path, file)
        if file[-4:] == 'json':
            TI_radar_json_path = os.path.join(ti_path, file)

    frame_num += 1
    TI_radar_points = read_pcd(TI_pcd_path)     # (N, 3)
    print('frame_num:', frame_num)
    print('TI_points: ', TI_radar_points.shape)
    print('img_path: ', img_path, '\n')
    TI_json_data = load_json(TI_radar_json_path)
    img_json_data = load_json(img_json)

    A = np.array(img_json_data['IntrinsicMatrix'])

    H = np.array(TI_json_data['TIRadar_to_LeopardCamera1_TransformMatrix'])
    B = np.dot(np.linalg.inv(A), H)

    xyz_radar = np.swapaxes(TI_radar_points, 0, 1)
    xyz1 = np.concatenate([xyz_radar, np.ones([1, xyz_radar.shape[1]])])  # 给xyz在后面叠加了一行1 (4,N)
    xyz_radar2 = np.dot(H, xyz1)  # (3,N)  相机坐标系下的xyz坐标
    xyz_radar2[0, :] = xyz_radar2[0, :] / xyz_radar2[2, :]
    xyz_radar2[1, :] = xyz_radar2[1, :] / xyz_radar2[2, :]
    # xyz_radar2 = xyz_radar2[2, :] / xyz_radar2[2, :]
    img = cv2.imread(img_path)

    for i in range(xyz_radar2.shape[1]):
        cv2.circle(img, (int(xyz_radar2[0, i]), int(xyz_radar2[1, i])), radius=10, color=(0, 0, 255), thickness=-1)

    save_name = '/home/zhangq/Desktop/zhangq/UsefulCode/' + img_path.split('/')[-1]
    # cv2.imwrite(save_name, img)
    cv2.imshow('img', cv2.resize(img, (1280, 720)))
    if cv2.waitKey(0) & 0xFF == 27:
        break

print('done')




