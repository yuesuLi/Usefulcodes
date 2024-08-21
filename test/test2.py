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

cmap = plt.cm.get_cmap('hsv', 256)
cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

def aaaa():
    print("aaaaaaaaaaaaaa")

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


def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
    return json_data


def ptsview(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    d = np.sqrt(x ** 2 + y ** 2)
    vals = 'height'
    if vals == 'height':
        col = z
    else:
        col = d
    # f = mayavi.mlab.gcf()
    # camera = f.scene.camera
    # camera.yaw(90)
    fig = mayavi.mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
    # camera = fig.scene.camera
    # camera.yaw(90)
    # cam, foc = mayavi.mlab.move()
    # print(cam, foc)
    mayavi.mlab.points3d(x, y, z,
                         col,
                         mode='point',
                         colormap='spectral',
                         figure=fig)
    mayavi.mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    axes = np.array(
        [[20, 0, 0, ], [0, 20, 0], [0, 0, 20]]
    )
    mayavi.mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig
    )
    mayavi.mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig
    )
    mayavi.mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig
    )
    mayavi.mlab.show()


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
                             pc_data[metadata['fields'][2]][:, None],
                             pc_data[metadata['fields'][3]][:, None]], axis=-1)
    # print(points.shape)

    if pts_view:
        ptsview(points)
    return points

def load_pcd_to_ndarray(pcd_path):
    with open(pcd_path) as f:
        while True:
            ln = f.readline().strip()
            if ln.startswith('DATA'):
                break

        points = np.loadtxt(f)
        points = points[:, 0:4]
        return points



# radar_points:(N,4)   radar_2d:(N,4) u,v,depth,velo
def get_uv_from_points(radar_points, TransformMatrix, width, height):
    # radar_points = radar_points.T
    velo = radar_points[:, 3:4]     # vel
    xyz1 = np.hstack((radar_points[:, :3], np.ones((radar_points.shape[0], 1)))) # 给xyz在后面叠加了一行1 (N,4)
    # xyz1 = np.concatenate([radar_points[:, :3], np.ones([1, radar_points.shape[1]])])
    radar_points_2d = np.dot(xyz1, np.transpose(TransformMatrix))

    # xyz1 = np.hstack((radar_points[:, :3], np.ones((radar_points.shape[0], 1)))).T  # 给xyz在后面叠加了一行1 (4,N)
    # radar_points_2d = np.dot(TransformMatrix, xyz1).T       # (N,3)

    radar_points_2d[:, 0] = radar_points_2d[:, 0] / radar_points_2d[:, 2]
    radar_points_2d[:, 1] = radar_points_2d[:, 1] / radar_points_2d[:, 2]
    # mask = (radar_points_2d[:, 0] < width) & (radar_points_2d[:, 1] < height) & \
    #        (radar_points_2d[:, 0] > 0) & (radar_points_2d[:, 1] > 0) & \
    #        (radar_points_2d[:, 2] > 5) & (radar_points_2d[:, 2] < 70)
    # radar_points_2d = radar_points_2d[mask, :]
    # velo = velo[mask, :]
    radar_2d = np.concatenate([radar_points_2d, velo], axis=-1)

    # uv1 = np.dot(TransformMatrix, xyz1)
    # uv1[0, :] = uv1[0, :] / uv1[2, :]
    # uv1[1, :] = uv1[1, :] / uv1[2, :]
    # uv = uv1[0:2, :]
    # uv = uv.T

    return radar_2d


def get_filelist(dir, FileList):
    newDir = dir
    filter_list = ['MEMS']
    if os.path.isfile(dir):
        FileList.append(dir)
        # FileList.append(os.path.basename(dir))    # only return filename

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            if s in filter_list:
                continue
            newDir = os.path.join(dir, s)
            get_filelist(newDir, FileList)
    return FileList


def roty(t):
    # Rotation about the y-axis.
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotz(t):
    # Rotation about the z-axis.
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def project_to_image(pts_3d, P):

    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def my_predictions_to_kitti_format(img_detections, TransformMatrix, img_size):

    predictions = []
    all_corners_2d = []
    for detections in img_detections:
        if detections is None:
            continue
        predictions.append([detections[0], detections[1], detections[2], detections[3], detections[4], detections[5], [detections[6]]])
        # Rescale boxes to original image
        # for x, y, z, l, w, h, ry in detections:
        #     predictions.append([x, y, z, l, w, h, ry])

    predictions = np.array(predictions)

    # predictions = np.array(predictions)

    # if predictions.shape[0]:
    #     predictions[:, 1:] = lidar_to_camera_box(predictions[:, 1:], out_TransformMatrix)

    # print('predictions:', type(predictions), '\n', predictions)
    for i, box_3d in enumerate(predictions):
        x, y, z, l, w, h, ry = box_3d
        # z = box_3d[1]
        # x = -box_3d[2]
        # y = -box_3d[3]
        # print('box_3d_', i, ': ', box_3d)
        # ry = -ry - np.pi / 2
        # ry = np.arctan2(math.sin(ry), math.cos(ry))

        # R = roty(ry)
        R = rotz(ry)

        # 3d bounding box corners
        # x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        # y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        # z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        z_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # rotate and translate 3d bounding box
        # corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d = np.dot(R, np.vstack([x_corners, z_corners, y_corners]))
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + x
        corners_3d[1, :] = corners_3d[1, :] + y
        corners_3d[2, :] = corners_3d[2, :] + z

        # print('corners_3d:\n', np.transpose(corners_3d))
        # TransformMatrix = np.hstack((In_TransformMatrix, np.zeros((3, 1))))
        # TransformMatrix[:, 3] = out_TransformMatrix[:, 3]
        # print('In_TransformMatrix', TransformMatrix.shape)
        corners_2d = project_to_image(np.transpose(corners_3d), TransformMatrix)
        # print('corners_2d:\n', corners_2d)
        all_corners_2d.append(corners_2d)


    return all_corners_2d


def get_lidar_annotation(lidar_json_path):

    with open(lidar_json_path) as f:
        lidar_json_data = json.load(f)
    x, y, z, l, w, h, alpha = [], [], [], [], [], [], []
    # print("json_data:", type(lidar_json_data['annotation'][0]['x']))
    for idx in range(len(lidar_json_data['annotation'])):
        if lidar_json_data['annotation'][idx]['x'] >= 75 or lidar_json_data['annotation'][idx]['x'] <= 0\
                or lidar_json_data['annotation'][idx]['y'] >= 75 or lidar_json_data['annotation'][idx]['y'] <= -75:
            continue
        x.append(lidar_json_data['annotation'][idx]['x'])
        y.append(lidar_json_data['annotation'][idx]['y'])
        z.append(lidar_json_data['annotation'][idx]['z'])
        l.append(lidar_json_data['annotation'][idx]['l'])
        w.append(lidar_json_data['annotation'][idx]['w'])
        h.append(lidar_json_data['annotation'][idx]['h'])
        alpha.append(lidar_json_data['annotation'][idx]['alpha'])


    # width = 429
    # height = 543
    # x = [(75 - tmp_x) / 75 * width for tmp_x in x]
    # y = [(tmp_y + 75) / 150 * height for tmp_y in y]
    # l = [tmp_l / 75 * width for tmp_l in l]
    # w = [tmp_w / 150 * height for tmp_w in w]
    return x, y, z, l, w, h, alpha

def draw_projected_box3d(image, qs, color=(255, 0, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def drawBox(img, x, y, w, h, color, yaw=0):
    left_top = np.array([x, y]).astype(int)
    right_bottom = np.array([x + w, y + h]).astype(int)
    cv.rectangle(img, left_top, right_bottom, color=color, thickness=2)

# base_path = '/media/ourDataset/v1.0_label/20211027_1_group0021_123frames_25labeled'
# base_path = '/media/ourDataset/v1.0_label/20211027_1_group0026_144frames_29labeled'
# base_path = '/media/ourDataset/v1.0_label/20211027_1_group0028_134frames_27labeled'
base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'
# base_path = '/media/ourDataset/v1.0_label/20211027_1_group0010_148frames_30labeled'
# base_path = './20211027_2_group0051_72frames_15labeled'
save_path = './save/test/'

# cv.namedWindow('imgs', 0)
# cv.resizeWindow('imgs', 1600, 900)

folders = os.listdir(base_path)
folders = sorted(folders)

# print('folders', folders)
frame_num = -1
for folder in folders:
    frame_num += 1
    # print('frame_num:', frame_num)
    if frame_num % 5 == 0:


        camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
        for file in os.listdir(camera_path):
            if file[-3:] == 'png':
                img_path = os.path.join(camera_path, file)
        lidar_path = os.path.join(base_path, folder, 'VelodyneLidar')
        print('lidar_path:', lidar_path)
        for file in os.listdir(lidar_path):
            if file[-3:] == 'pcd':
                Lidar_pcd_path = os.path.join(lidar_path, file)
            if file[-4:] == 'json':
                Lidar_json_path = os.path.join(lidar_path, file)


        img = cv.imread(img_path)
        width, height = img.shape[1], img.shape[0]

        lidar_img = cv.imread(img_path)
        lidar_json_data = load_json(Lidar_json_path)
        matrix_lidar = np.array(lidar_json_data['VelodyneLidar_to_LeopardCamera1_TransformMatrix']) #3*4
        x, y, z, l, w, h, alpha = get_lidar_annotation(Lidar_json_path)
        # lidar_pts = read_pcd(Lidar_pcd_path)

        # print('lidar_annotation:', lidar_annotation)
        img_detections = []
        detection = []
        for i in range(len(x)):
            detection = [x[i], y[i], z[i], l[i], w[i], h[i], alpha[i]]
            img_detections.append(detection)

        # print('img_detections:', img_detections)
        img_size = [width, height] # width, height
        all_bbox_2d = my_predictions_to_kitti_format(img_detections, matrix_lidar, img_size)

        for bbox_2d in all_bbox_2d:
            img = draw_projected_box3d(img, bbox_2d)

        cv.imshow('img', cv.resize(img, (1080, 720)))
        cv.waitKey(0)





