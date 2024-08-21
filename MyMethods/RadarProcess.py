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

# uv:(N,4)
def draw_points_to_img(uv, img, size=4, num_clusters=1, db=None, colors=None):
    points = np.array((uv[:, 0].astype(int), uv[:, 1].astype(int))).T  # (N, 2)
    # colors = get_colors('RdBu', num_clusters)
    # colors = np.random.rand(num_clusters, 3)*255
    if num_clusters == 0:
        return

    # if db != None:
        # colors = colors*255
        # mymap = plt.cm.get_cmap('hsv', num_clusters)
        # mymap = np.array([mymap(i) for i in range(num_clusters)])[:, :3] * 255

    # for i in range(points.shape[0]):
    #     if db != None:
    #         if db.labels_[i] != -1:
    #             # color = colors[db.labels_[i]]
    #             # color = mymap[int(db.labels_[i]), :]
    #             color = cmap[int(np.floor(255/num_clusters) * db.labels_[i]), :]
    #             cv.circle(img, (points[i, 0], points[i, 1]), size,
    #                       color=tuple(color), thickness=-1)   # (color[0].item(), color[1].item(), color[2].item())
    #     else:
    #         depth = uv[i, 2]
    #         color = cmap[int(3 * depth), :]
    #         # color = np.squeeze(colors)
    #         # print('color:', color)
    #         cv.circle(img, (points[i, 0], points[i, 1]), size,
    #                   color=tuple(color), thickness=-1)


    if db == None:
        for i in range(points.shape[0]):
            depth = uv[i, 2]
            color = cmap[int(3 * depth), :]
            cv.circle(img, (points[i, 0], points[i, 1]), size, color=tuple(color), thickness=-1)
        return
    color_list = plt.cm.tab20(np.linspace(0, 1, num_clusters))*255
    for i in range(num_clusters):
        points_class = points[db.labels_ == i, :]
        color = color_list[i]
        # color = cmap[int(np.floor(255/num_clusters) * db.labels_[i]), :]
        for j in range(points_class.shape[0]):
            cv.circle(img, (points_class[j, 0].astype(int), points_class[j, 1].astype(int)), size, color=tuple(color), thickness=-1)


def draw_points(points, radar_type=None, order=1, num_clusters=1, db=None, colors=None, point_size=1):
    plt.subplot(2, 2, order)
    if num_clusters == 0:
        return
    # colors = np.random.rand(num_clusters, 3)  # used only for display
    # print('colours:', colours)
    flag = 1
    if radar_type == 'OCU':
        flag = 2
        plt.xlim((-25, 25))
        plt.ylim((0, 70))
    elif radar_type == 'TI':
        flag = 1
        plt.xlim((-20, 20))
        plt.ylim((0, 50))
        # point_size = 10

    if db == None:
        plt.scatter(points[:, 0], points[:, flag], point_size)
        return
    color_list = plt.cm.tab20(np.linspace(0, 1, num_clusters))
    point_size = 5
    for i in range(num_clusters):
        points_class = points[db.labels_ == i, :]
        # color = cmap[int(np.floor(255/num_clusters) * db.labels_[i]), :]/255
        color = color_list[i]
        plt.scatter(points_class[:, 0], points_class[:, flag], point_size, color=tuple(color))


    # if db != None:
    #     point_size = 5
    #     for i in range(points.shape[0]):
    #         if db.labels_[i] != -1:
    #             plt.scatter(points[i, 0], points[i, flag], point_size, color=colors[db.labels_[i], :])
    # else:
    #     plt.scatter(points[:, 0], points[:, flag], point_size)

    # plt.axis('scaled')  # {equal, scaled}
    plt.xlabel('X')
    plt.ylabel('Y')


def draw_points_ave_to_img(uv, img, size=4, num_clusters=1, db=None, colors=None):
    points = np.array((uv[:, 0].astype(int), uv[:, 1].astype(int))).T  # (N, 2)
    # color = cmap[int(3 * depth), :]
    if num_clusters == 0:
        return
    if num_clusters != 1:
        colors = colors*255

    if db == None:
        for i in range(points.shape[0]):
            depth = uv[i, 2]
            color = cmap[int(3 * depth), :]
            cv.circle(img, (points[i, 0], points[i, 1]), size, color=tuple(color), thickness=-1)
        return
    size = 15
    for i in range(num_clusters):
        xy = points[db.labels_ == i]
        color = colors[i]
        xy_ave = np.array([np.average(xy[:, 0]), np.average(xy[:, 1])])
        cv.circle(img, (xy_ave[0].astype(int), xy_ave[1].astype(int)), size, (color[0].item(), color[1].item(), color[2].item()), thickness=-1)


def draw_points_ave(points, radar_type=None, order=1, num_clusters=1, db=None, colors=None, point_size=10):
    plt.subplot(2, 2, order)
    if num_clusters == 0:
        return
    flag = 1
    if radar_type == 'OCU':
        flag = 2
        plt.xlim((-50, 50))
        plt.ylim((0, 70))
        # point_size = 30
    elif radar_type == 'TI':
        flag = 1
        plt.xlim((-40, 40))
        plt.ylim((0, 50))
        # point_size = 30

    if db != None:
        point_size = 40
        for i in range(num_clusters):
            xy = points[db.labels_ == i]
            xy_ave = np.array([np.average(xy[:, 0]), np.average(xy[:, flag])])
            plt.scatter(xy_ave[0], xy_ave[1], point_size, color=colors[i, :])

    else:
        plt.scatter(points[:, 0], points[:, flag], point_size)

    plt.xlabel('X')
    plt.ylabel('Y')

def Density_peak():

    print()

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

# radar_uvdv:(N,4), radar_value(w,h,2), lidar_value(w,h)
def uv_filter(radar_uvdv, radar_value, lidar_value, width, height):
    mask = radar_uvdv[:, 2] > 1e6
    # raw_radar_uvdv = radar_uvdv
    # radar_value:(w,h,3)
    radar_value = np.concatenate((radar_value, np.zeros((radar_value.shape[0], radar_value.shape[1], 1))), axis=2)
    for i in range(radar_uvdv.shape[0]):
        depth = radar_uvdv[i, 2]
        velo = radar_uvdv[i, 3]
        # color = cmap[int(3 * depth), :]
        # cv2.circle(radar_img,
        #            (int(np.round(radar_points[i, 0])), int(np.round(radar_points[i, 1]))),
        #            3, color=tuple(color), thickness=-1)
        radar_value[int(np.floor(radar_uvdv[i, 0])), int(np.floor(radar_uvdv[i, 1])), 0] = depth
        radar_value[int(np.floor(radar_uvdv[i, 0])), int(np.floor(radar_uvdv[i, 1])), 1] = velo
        radar_value[int(np.floor(radar_uvdv[i, 0])), int(np.floor(radar_uvdv[i, 1])), 2] = i

    for u in range(radar_value.shape[0]):
        for v in range(radar_value.shape[1]):
            if radar_value[u, v, 0] != 0:  # depth
                # diff = []
                save_flag = False
                # update_flag = False
                for i in range(u - 10, u + 10):
                    for j in range(v - 10, v + 10):
                        if i >= width or j >= height or i < 0 or j < 0:
                            continue
                        # if lidar_value[i, j] != 0:
                        #     diff.append((lidar_value[i, j] - radar_value[u, v, 0]))
                        if (abs(lidar_value[i, j] - radar_value[u, v, 0]) <= radar_value[u, v, 1]):
                            save_flag = True

                if save_flag:
                    mask[int(radar_value[u, v, 2])] = True
                    # print('radar_value[u, v, 2]:', type(int(radar_value[u, v, 2])), int(radar_value[u, v, 2]))
    print()
    radar_uvdv = radar_uvdv[mask, :]
    return radar_uvdv

# radar_points:(N,4) xyzv;   radar_value(w,h,2),
def points_filter(radar_points, radar_value, radar_type, width, height):

    if radar_type =='OCU':
        mask = (radar_points[:, 2] < 70) & (radar_points[:, 2] > 5)
    elif radar_type =='TI':
        mask = (radar_points[:, 1] < 70) & (radar_points[:, 1] > 5)

    radar_points = radar_points[mask, :]
    raw_radar_points = np.zeros((radar_points.shape[0], radar_points.shape[1]))
    radar_points = raw_radar_points
    # distance_matrix =
    for i in range(radar_points):
        save_flag = False
        for j in range(u - 10, u + 10):
            for j in range(v - 10, v + 10):
                if i >= width or j >= height or i < 0 or j < 0:
                    continue
                # if lidar_value[i, j] != 0:
                #     diff.append((lidar_value[i, j] - radar_value[u, v, 0]))
                if abs(lidar_value[i, j] - radar_value[u, v, 0]) <= radar_value[u, v, 1]:
                    save_flag = True

        if save_flag:
            mask[int(radar_value[u, v, 2])] = True
            # print('radar_value[u, v, 2]:', type(int(radar_value[u, v, 2])), int(radar_value[u, v, 2]))


    mask = radar_points[:, 3] > 1e6

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
frame_num = -1
for folder in folders:
    frame_num += 1
    print('frame_num:', frame_num)
    if frame_num == 1:


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
        # lidar_img = np.zeros((height, width, 3), dtype=np.uint8)
        lidar_value = np.zeros((width, height), dtype=np.float32)
        radar_img = np.zeros((height, width, 3), dtype=np.uint8)
        radar_value = np.zeros((width, height, 2), dtype=np.float32)

        lidar_img = cv.imread(img_path)
        lidar_json_data = load_json(Lidar_json_path)
        matrix_lidar = np.array(lidar_json_data['VelodyneLidar_to_LeopardCamera1_TransformMatrix'])
        lidar_annotation = lidar_json_data['annotation']
        lidar_pts = read_pcd(Lidar_pcd_path)

        # np.set_printoptions(threshold=np.inf)
        print('lidar_pts', lidar_pts)

        #
        # n = lidar_pts.shape[0]
        # lidar_pts = np.hstack((lidar_pts[:, :3], np.ones((n, 1))))
        # lidar_2d = np.dot(lidar_pts, np.transpose(matrix_lidar))
        # lidar_2d[:, 0] = lidar_2d[:, 0] / lidar_2d[:, 2]
        # lidar_2d[:, 1] = lidar_2d[:, 1] / lidar_2d[:, 2]
        # mask = (lidar_2d[:, 0] < width) & (lidar_2d[:, 1] < height) & \
        #        (lidar_2d[:, 0] > 0) & (lidar_2d[:, 1] > 0) & \
        #        (lidar_2d[:, 2] > 5) & (lidar_2d[:, 2] < 80)
        # lidar_2d = lidar_2d[mask, :]        # lidar_2d: [:, u,v,depth]
        #
        # print('lidar_2d:', lidar_2d.shape)
        # cmap = plt.cm.get_cmap('hsv', 256)
        # cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        # for i in range(lidar_2d.shape[0]):
        #     depth = lidar_2d[i, 2]
        #     # depth = pts_2d[i,2]
        #     color = cmap[int(3 * depth), :]
            # cv.circle(lidar_img,
            #            (int(np.round(lidar_2d[i, 0])), int(np.round(lidar_2d[i, 1]))),
            #            3, color=tuple(color), thickness=-1)
            # lidar_value[int(np.floor(lidar_2d[i, 0])), int(np.floor(lidar_2d[i, 1]))] = depth
        # Image.fromarray(lidar_img).show()



        raw_img = cv.imread(img_path)
        OCU_img = cv.imread(img_path)
        TI_img = cv.imread(img_path)
        OCU_img_dbscan = cv.imread(img_path)
        TI_img_dbscan = cv.imread(img_path)


        OCU_radar_points = load_pcd_to_ndarray(OCU_pcd_path)    # (N,4)
        print('OCU_radar_points:', OCU_radar_points)
        mask = (OCU_radar_points[:, 2] < 70) & (OCU_radar_points[:, 2] > 5)
        OCU_radar_points = OCU_radar_points[mask, :]

        TI_radar_points = load_pcd_to_ndarray(TI_pcd_path)
        print('TI_radar_points:', TI_radar_points)
        mask = (TI_radar_points[:, 1] < 70) & (TI_radar_points[:, 1] > 5)
        TI_radar_points = TI_radar_points[mask, :]


        print('OCU_points: ', OCU_radar_points.shape)
        print('TI_points: ', TI_radar_points.shape)
        #
        OCU_radar_points_xyv = np.concatenate((OCU_radar_points[:, 0:1], OCU_radar_points[:, 2:3], OCU_radar_points[:, 3:4]*0.5), axis=1)
        TI_radar_points_xyv = np.concatenate((TI_radar_points[:, 0:2], TI_radar_points[:, 3:4]*0.5), axis=1)

        # db.labels_是一个形为(N, )的ndarray，N为点的数量，数组内值为各点所属类别，类别数从0开始，-1代表噪声不算入类别数, math.ceil(OCU_radar_points.shape[0]/20)
        # print('TI_db.labels_', TI_db.labels_.shape, type(TI_db.labels_), TI_db.labels_)
        # OCU_num = sorted([10, np.ceil(OCU_radar_points.shape[0]/100), 15])[1]
        # TI_num = sorted([10, np.ceil(TI_radar_points.shape[0]/50), 15])[1]
        # print('OCU_num:', OCU_num)
        # print('TI_num:', TI_num)
        OCU_db = DBSCAN(eps=1.5, min_samples=10).fit(OCU_radar_points_xyv)
        TI_db = DBSCAN(eps=1.5, min_samples=10).fit(TI_radar_points_xyv)


        OCU_dbscan_points = OCU_radar_points[OCU_db.labels_[:] != -1]
        TI_dbscan_points = TI_radar_points[TI_db.labels_[:] != -1]
        print('OCU_dbscan_points: ', OCU_dbscan_points.shape)
        print('TI_dbscan_points: ', TI_dbscan_points.shape)

        OCU_num_dbscan = len(np.unique(OCU_db.labels_)) - (1 if -1 in OCU_db.labels_ else 0)
        print("OCU_num_dbscan: ", OCU_num_dbscan)
        TI_num_dbscan = len(np.unique(TI_db.labels_)) - (1 if -1 in TI_db.labels_ else 0)
        print("TI_num_dbscan: ", TI_num_dbscan)


        OCU_colors = np.random.rand(OCU_num_dbscan, 3)
        TI_colors = np.random.rand(TI_num_dbscan, 3)

        plt.suptitle(str(frame_num) + '_BEV radar points')        # main title
        draw_points(OCU_radar_points, 'OCU', 1)
        plt.title('OCU_raw')
        draw_points(TI_radar_points, 'TI', 2)
        plt.title('TI_raw')
        draw_points(OCU_radar_points, 'OCU', 3, OCU_num_dbscan, OCU_db, OCU_colors)
        plt.title('OCU_dbscan')
        draw_points(TI_radar_points, 'TI', 4, TI_num_dbscan, TI_db, TI_colors)
        plt.title('TI_dbscan')
        plt.subplots_adjust(hspace=0.5, wspace=0.5)     #
        # plt.savefig('./BEV_points.jpg')
        BEV_savename = save_path + str(frame_num) + '_BEV_points.jpg'
        plt.savefig(BEV_savename)
        plt.clf()
        # plt.show()

        OCU_radar_json_data = load_json(OCU_radar_json_path)
        OCU_TransformMatrix = np.array(OCU_radar_json_data['OCULiiRadar_to_LeopardCamera1_TransformMatrix'])
        TI_radar_json_data = load_json(TI_radar_json_path)
        TI_TransformMatrix = np.array(TI_radar_json_data['TIRadar_to_LeopardCamera1_TransformMatrix'])


        # get_uv_from_points   return(N,4)  u,v,depth,velo
        OCU_uv = get_uv_from_points(OCU_radar_points, OCU_TransformMatrix, width, height)
        TI_uv = get_uv_from_points(TI_radar_points, TI_TransformMatrix, width, height)
        # OCU_uv_dbscan = get_uv_from_points(OCU_radar_points, OCU_TransformMatrix)
        # TI_uv_dbscan = get_uv_from_points(TI_radar_points, TI_TransformMatrix)
        # OCU_radar_value = np.zeros((width, height, 2), dtype=np.float32)
        # TI_radar_value = np.zeros((width, height, 2), dtype=np.float32)
        # # print('OCU_uv:', OCU_uv.shape,'\n', OCU_uv)
        # print('OCU_uv:', OCU_uv.shape)
        # print('TI_uv:', TI_uv.shape)
        #
        # OCU_uv_filter = uv_filter(OCU_uv, OCU_radar_value, lidar_value, width, height)
        # TI_uv_filter = uv_filter(TI_uv, TI_radar_value, lidar_value, width, height)
        #
        # print('OCU_uv_filter:', OCU_uv_filter.shape)
        # print('TI_uv_filter:', TI_uv_filter.shape)

        OCU_size = 5
        TI_size = 5
        draw_points_to_img(OCU_uv, OCU_img, OCU_size)
        draw_points_to_img(TI_uv, TI_img, TI_size)
        # draw_points_to_img(OCU_uv_filter, OCU_img_dbscan, 10)
        # draw_points_to_img(TI_uv_filter, TI_img_dbscan, 10)
        draw_points_to_img(OCU_uv, OCU_img_dbscan, 10, OCU_num_dbscan, OCU_db, OCU_colors)
        draw_points_to_img(TI_uv, TI_img_dbscan, 10, TI_num_dbscan, TI_db, TI_colors)
        # draw_points_ave_to_img


        # cv.namedWindow('lidar_img', 0)
        # cv.resizeWindow('lidar_img', 1500, 720)
        # lidar_savename = save_path + str(frame_num) + '_lidar_img.jpg'
        # cv.imwrite(lidar_savename, lidar_img)
        # cv.imshow('lidar_img', lidar_img)
        # cv.waitKey(0)

        imgs1 = np.hstack([OCU_img, TI_img])
        imgs2 = np.hstack([OCU_img_dbscan, TI_img_dbscan])
        imgs = np.vstack([imgs1, imgs2])

        # cv.imshow('imgs', imgs)
        # cv.waitKey(0)

        points_savename = save_path + str(frame_num) + '_points_on_img.jpg'
        cv.imwrite(points_savename, imgs)
        cv.imwrite('points_on_img.jpg', imgs)

        print()





# def draw_points(points, fig, radar_type = None, order = 1,point_size = 10):
#     ax = fig.add_subplot(2, 2, order)
#     color = [1, 0, 0]
#     flag = 1
#     title = 'Point Cloud'
#     if radar_type == 'OCU':
#         flag = 2
#         title = 'OCU Point Cloud'
#     elif radar_type == 'TI':
#         flag = 1
#         title = 'TI Point Cloud'
#     ax.scatter(points[:, 0], points[:, flag], point_size)
#     # plt.title(title)
#     ax.axis('scaled')  # {equal, scaled}
#     ax.set_xlabel('X')
#     ax.set_ylabel('Z')
#     # ax.set_zlabel('Z')
#     # plt.show()



# OURDATASET_ROOT = "/media/ourDataset/v1.0"
# dayExp = "20211027_2"
# File_List = []
# File_List = get_filelist(raw_path, File_List)
# print(File_List)
# os.path.join(OURDATASET_ROOT, dayExp, "Dataset")


# img_path = './20211027_2_group0051_72frames_15labeled/20211027_2_group0051_frame0015_labeled/LeopardCamera1/1635327418.991.png'
# image_json_path = './20211027_2_group0051_72frames_15labeled/20211027_2_group0051_frame0015_labeled/LeopardCamera1/1635327418.991.json'
#
# OCU_pcd_path = './20211027_2_group0051_72frames_15labeled/20211027_2_group0051_frame0015_labeled/OCULiiRadar/1635327421.504.pcd'
# OCU_radar_json_path = './20211027_2_group0051_72frames_15labeled/20211027_2_group0051_frame0015_labeled/OCULiiRadar/1635327421.504.json'
#
# TI_pcd_path = './20211027_2_group0051_72frames_15labeled/20211027_2_group0051_frame0015_labeled/TIRadar/1635327418.496.pcd'
# TI_radar_json_path = './20211027_2_group0051_72frames_15labeled/20211027_2_group0051_frame0015_labeled/TIRadar/1635327418.496.json'


