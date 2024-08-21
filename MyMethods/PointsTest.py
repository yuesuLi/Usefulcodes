import argparse
import os
import sys
from pathlib import Path
import numpy as np
import json

import re
import mayavi
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import copy

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


def pointcloud_transform(pointcloud, transform_matrix):
    '''
        transform pointcloud from coordinate1 to coordinate2 according to transform_matrix
    :param pointcloud: (x, y, z, ...)
    :param transform_matrix:
    :return pointcloud_transformed: (x, y, z, ...)
    '''
    n_points = pointcloud.shape[0]
    xyz = pointcloud[:, :3]
    xyz1 = np.vstack((xyz.T, np.ones((1, n_points))))
    xyz1_transformed = np.matmul(transform_matrix, xyz1)
    pointcloud_transformed = np.hstack((
        xyz1_transformed[:3, :].T,
        pointcloud[:, 3:]
    ))
    return pointcloud_transformed

def pts2rbev(lidar_json, TI_json, cam_json):

    # Get LiDAR annotation points
    lidar_raw_anno = lidar_json['annotation']
    x, y, z = [], [], []
    for idx in range(len(lidar_raw_anno)):
        if lidar_raw_anno[idx]['x'] >= 50 or lidar_raw_anno[idx]['x'] <= 0 \
                or lidar_raw_anno[idx]['y'] >= 25 or lidar_raw_anno[idx]['y'] <= -25:
            continue
        x.append(lidar_raw_anno[idx]['x'])
        y.append(lidar_raw_anno[idx]['y'])
        z.append(lidar_raw_anno[idx]['z'])

    if len(x) > 0:
        Lidar_points = np.vstack((x, y, z)).T   # (N, 3)
    else:
        Lidar_points = np.array([])
        return Lidar_points


    # LiDAR points to radar coordinate
    VelodyneLidar_to_LeopardCamera1_TransformMatrix = np.array(lidar_json['VelodyneLidar_to_LeopardCamera1_TransformMatrix'])
    TIRadar_to_LeopardCamera1_TransformMatrix = np.array(TI_json['TIRadar_to_LeopardCamera1_TransformMatrix'])
    LeopardCamera1_IntrinsicMatrix = np.array(cam_json['IntrinsicMatrix'])
    VelodyneLidar_to_TIRadar_TransformMatrix = np.matmul(
        np.linalg.inv(
            np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                                 TIRadar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
        ),
        np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                             VelodyneLidar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
    )

    Lidar_points = pointcloud_transform(Lidar_points, VelodyneLidar_to_TIRadar_TransformMatrix)

    return Lidar_points




base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'

folders = os.listdir(base_path)
folders = sorted(folders)
frame_num = -1
plot = True
for folder in folders:
    if 'labeled' not in folder:
        continue
    # camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
    # for file in os.listdir(camera_path):
    #     if file[-3:] == 'png':
    #         img_path = os.path.join(camera_path, file)
    #     if file[-4:] == 'json':
    #         img_json = os.path.join(camera_path, file)
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
            TI_json_path = os.path.join(ti_path, file)

    frame_num += 1
    TI_radar_points = read_pcd(TI_pcd_path)     # (N, 3)
    length = TI_radar_points.shape[0]
    a = TI_radar_points[:, 0]
    b = TI_radar_points[:, 1]
    c = TI_radar_points[:, 2]

    points2 = np.concatenate([TI_radar_points[:, 0].reshape((length, 1)),
                              TI_radar_points[:, 2].reshape((length, 1)),
                              TI_radar_points[:, 1].reshape((length, 1))], axis=1)
    # lidar_json = load_json(calib_lidar)
    # TI_json = load_json(TI_json_path)
    # cam_json = load_json(img_json)

    # lidar2TI_anno = get_lidar2TI_anno(lidar_json, TI_json)
    # print('TI_pcd_path:', TI_pcd_path)
    # print('img_path:', img_path)
    print('TI_points: ', TI_radar_points.shape, '\n')




