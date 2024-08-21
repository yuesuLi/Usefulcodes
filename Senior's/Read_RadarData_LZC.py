# Semantic-guided depth completion
import open3d as o3d
import numpy as np
import mayavi.mlab
import cv2
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
from pypcd.pypcd import PointCloud
import re
import warnings

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
    print(points.shape)

    if pts_view:
        ptsview(points)
    return points


def pts2camera(pts, img_path, calib_path):
    img = Image.open(img_path)
    print(f'img: {img.size}')
    width, height = img.size
    try:
        matrix = json.load(open(calib_path))['VelodyneLidar_to_LeopardCamera1_TransformMatrix']
    except:
        matrix = json.load(open(calib_path))['OCULiiRadar_to_LeopardCamera1_TransformMatrix']
    matrix = np.asarray(matrix)
    n = pts.shape[0]
    pts = np.hstack((pts, np.ones((n, 1))))
    pts_2d = np.dot(pts, np.transpose(matrix))
    pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2]
    pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2]
    mask = (pts_2d[:, 0] < width) & (pts_2d[:, 1] < height) & (pts_2d[:, 2] > 5) & (pts_2d[:, 2] < 80)
    pts_2d = pts_2d[mask, :]
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    img = np.asarray(img)
    for i in range(pts_2d.shape[0]):
        depth = pts_2d[i, 2]
        # depth = pts_2d[i,2]
        color = cmap[int(3 * depth), :]
        cv2.circle(img, (int(np.round(pts_2d[i, 0])),
                         int(np.round(pts_2d[i, 1]))),
                   3, color=tuple(color), thickness=-1)

    save_path = os.path.join('./output80m', img_path.split('/')[-1])
    img = Image.fromarray(img)
    # img.save(save_path)
    img.show()
    return img


def pts2bev(pts):
    side_range = (-40, 40)
    fwd_range = (0, 80)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    f_filter = np.logical_and(x > fwd_range[0], x < fwd_range[1])
    s_filter = np.logical_and(y > side_range[0], y < side_range[1])
    filter = np.logical_and(f_filter, s_filter)
    indices = np.argwhere(filter).flatten()
    x, y, z = x[indices], y[indices], z[indices]

    res = 0.5
    x_img = (-y / res).astype(np.int32)
    y_img = (-x / res).astype(np.int32)
    x_img = x_img - int(np.floor(side_range[0]) / res)
    y_img = y_img + int(np.floor(fwd_range[1]) / res)

    height_range = (-2, 0.5)
    pixel_value = np.clip(a=z, a_max=height_range[1], a_min=height_range[0])

    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)

    pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])

    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)

    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value

    im = Image.fromarray(im)
    im.show()

    return im


def lidar_radar_diff(img_path, lidar_pts, radar_pts, calib_lidar, calib_radar):
    img = Image.open(img_path)
    width, height = img.size
    lidar_img = np.zeros((height, width, 3), dtype=np.uint8)
    lidar_value = np.zeros((width, height), dtype=np.float32)
    radar_img = np.zeros((height, width, 3), dtype=np.uint8)
    radar_value = np.zeros((width, height, 2), dtype=np.float32)

    matrix_lidar = json.load(open(calib_lidar))['VelodyneLidar_to_LeopardCamera1_TransformMatrix']
    try:
        matrix_radar = json.load(open(calib_radar))['OCULiiRadar_to_LeopardCamera1_TransformMatrix']
    except:
        matrix_radar = json.load(open(calib_ti))['TIRadar_to_LeopardCamera1_TransformMatrix']
    matrix_lidar = np.asarray(matrix_lidar)
    matrix_radar = np.asarray(matrix_radar)

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    n = lidar_pts.shape[0]
    lidar_pts = np.hstack((lidar_pts[:, :3], np.ones((n, 1))))
    lidar_2d = np.dot(lidar_pts, np.transpose(matrix_lidar))
    lidar_2d[:, 0] = lidar_2d[:, 0] / lidar_2d[:, 2]
    lidar_2d[:, 1] = lidar_2d[:, 1] / lidar_2d[:, 2]
    mask = (lidar_2d[:, 0] < width) & (lidar_2d[:, 1] < height) & \
           (lidar_2d[:, 0] > 0) & (lidar_2d[:, 1] > 0) & \
           (lidar_2d[:, 2] > 5) & (lidar_2d[:, 2] < 80)
    lidar_2d = lidar_2d[mask, :]

    for i in range(lidar_2d.shape[0]):
        depth = lidar_2d[i, 2]
        # depth = pts_2d[i,2]
        color = cmap[int(3 * depth), :]
        cv2.circle(lidar_img,
                   (int(np.round(lidar_2d[i, 0])), int(np.round(lidar_2d[i, 1]))),
                   3, color=tuple(color), thickness=-1)
        lidar_value[int(np.floor(lidar_2d[i, 0])), int(np.floor(lidar_2d[i, 1]))] = depth
    Image.fromarray(lidar_img).show()

    n = radar_pts.shape[0]
    velo = radar_pts[:, 3:4]
    radar_pts = np.hstack((radar_pts[:, :3], np.ones((n, 1))))
    radar_2d = np.dot(radar_pts, np.transpose(matrix_radar))
    radar_2d[:, 0] = radar_2d[:, 0] / radar_2d[:, 2]
    radar_2d[:, 1] = radar_2d[:, 1] / radar_2d[:, 2]
    mask = (radar_2d[:, 0] < width) & (radar_2d[:, 1] < height) & \
           (radar_2d[:, 0] > 0) & (radar_2d[:, 1] > 0) & \
           (radar_2d[:, 2] > 5) & (radar_2d[:, 2] < 80)
    radar_2d = radar_2d[mask, :]
    velo = velo[mask, :]
    radar_2d = np.concatenate([radar_2d, velo], axis=-1)

    print(f'radar_2d number: {radar_2d.shape[0]}, lidar_2d number: {lidar_2d.shape[0]}')
    for i in range(radar_2d.shape[0]):
        depth = radar_2d[i, 2]
        velo = radar_2d[i, 3]
        color = cmap[int(3 * depth), :]
        cv2.circle(radar_img,
                   (int(np.round(radar_2d[i, 0])), int(np.round(radar_2d[i, 1]))),
                   3, color=tuple(color), thickness=-1)
        # print(int(np.round(radar_2d[i, 0])), int(np.round(radar_2d[i, 1])))
        radar_value[int(np.floor(radar_2d[i, 0])), int(np.floor(radar_2d[i, 1])), 0] = depth
        radar_value[int(np.floor(radar_2d[i, 0])), int(np.floor(radar_2d[i, 1])), 1] = velo
    Image.fromarray(radar_img).show()
    # print(radar_value.shape)
    radar_list, min_list, velo_list, max_list = [], [], [], []
    for u in range(radar_value.shape[0]):
        for v in range(radar_value.shape[1]):
            if radar_value[u, v, 0] != 0:
                # print(f'radar depth: {radar_value[u, v, 0]}, radar velo: {radar_value[u, v, 1]}')
                diff = []
                for i in range(u - 6, u + 6):
                    for j in range(v - 6, v + 6):
                        if i > width or j > height:
                            continue
                        if lidar_value[i, j] != 0:
                            diff.append((lidar_value[i, j] - radar_value[u, v, 0]))
                            # print(lidar_value[i, j], end=' ')
                if len(diff) > 0:
                    # print(f'min: {min(diff)}, max: {max(diff)}, mean: {sum(diff)/len(diff)}')
                    min_list.append(radar_value[u, v, 0] + min(diff))
                    radar_list.append(radar_value[u, v, 0])
                    max_list.append(radar_value[u, v, 0] + max(diff))
                    velo_list.append(radar_value[u, v, 1])
    idx = sorted(range(len(radar_list)), key=lambda k: radar_list[k])
    radar_list = np.asarray(radar_list)[idx]
    min_list = np.asarray(min_list)[idx]
    max_list = np.asarray(max_list)[idx]
    velo_list = np.asarray(velo_list)[idx]

    plt.plot(radar_list, radar_list + velo_list, '--', linewidth=1, color='sandybrown')
    plt.plot(radar_list, radar_list - velo_list, '--', linewidth=1, color='sandybrown')
    plt.fill_between(radar_list, radar_list, radar_list - velo_list, color='peachpuff')
    plt.fill_between(radar_list, radar_list, radar_list + velo_list, color='indianred')

    plt.plot(radar_list, radar_list, '--', linewidth=1, color='darkcyan')
    plt.scatter(radar_list, min_list, marker='v', c='darkcyan', s=24)
    plt.scatter(radar_list, max_list, marker='^', c='darkcyan', s=24)

    for i in range(len(radar_list)):
        plt.plot([radar_list[i], radar_list[i]], [min_list[i], max_list[i]], '-', linewidth=1, color='darkcyan')
        # plt.plot(radar_list[i], max_list[i], 'b-', linewidth=1)
    # print(radar_list[:5], lidar_list[:5], err_list[:5])
    # plt.errorbar(radar_list, lidar_list, yerr=err_list, label='None')
    plt.show()


if __name__ == '__main__':
    folders = os.listdir('./Dataset')
    folders = sorted(folders)
    for folder in folders:
        camera_path = os.path.join('./Dataset', folder, 'LeopardCamera1')
        for file in os.listdir(camera_path):
            if file[-3:] == 'png':
                img_path = os.path.join(camera_path, file)
        lidar_path = os.path.join('./Dataset', folder, 'VelodyneLidar')
        for file in os.listdir(lidar_path):
            if file[-3:] == 'pcd':
                pcd_lidar = os.path.join(lidar_path, file)
            if file[-4:] == 'json':
                calib_lidar = os.path.join(lidar_path, file)
        radar_path = os.path.join('./Dataset', folder, 'OCULiiRadar')
        for file in os.listdir(radar_path):
            if file[-3:] == 'pcd':
                pcd_radar = os.path.join(radar_path, file)
            if file[-4:] == 'json':
                calib_radar = os.path.join(radar_path, file)

        ti_path = os.path.join('./Dataset', folder, 'TIRadar')
        for file in os.listdir(ti_path):
            if file[-3:] == 'pcd':
                pcd_ti = os.path.join(ti_path, file)
            if file[-4:] == 'json':
                calib_ti = os.path.join(ti_path, file)
        # pts = read_pcd(pcd_lidar, False)
        # pts = read_pcd(pcd_radar, False)
        # img = pts2camera(pts[:, :3], img_path, calib_radar)
        # img = pts2bev(pts[:, :3])
        lidar_radar_diff(img_path, read_pcd(pcd_lidar), read_pcd(pcd_radar), calib_lidar, calib_radar)
        lidar_radar_diff(img_path, read_pcd(pcd_lidar), read_pcd(pcd_ti), calib_lidar, calib_ti)
        break
