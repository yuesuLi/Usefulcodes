import json
import cv2 as cv
import numpy as np
import math
import os
import struct
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R

def load_TIRadarPcd(pcd_path):
    '''
    read TI radar pcd
    :param pcd_path: str - TI radar pcd path
    :return: np.array-(n, 5) x,y,z,velocity,snr
    '''
    with open(pcd_path, "r") as f:
        data = f.readlines()
    data = data[10:]
    if len(data) == 0:  # when TIRadar pointcloud have no points
        return np.zeros((1,5))
    data = np.array([[float(point_attribute) for point_attribute in point_row.split()] for point_row in data])
    return data

def load_VelodyneLidarPcd(pcd_path):
    '''
    read velodyne lidar pcd
    :param pcd_path: str-velodyne lidar pcd path
    :return: np.array-(n, 6) x,y,z,intensity,ring,time
    '''
    with open(pcd_path, "rb") as f:
        f.seek(212, 0)
        data = f.read()
    num_points = math.floor(len(data) / 22)
    data_valid_len = num_points * 22
    data = struct.unpack('<' + 'ffffhf' * num_points, data[:data_valid_len])
    data = np.array(data).reshape((-1, 6))
    return data

def draw_2d_box(ax, annotation_2d):
    xs = [annotation_2d['x'], annotation_2d['x'], annotation_2d['x']+annotation_2d['w'], annotation_2d['x']+annotation_2d['w'], annotation_2d['x']]
    ys = [annotation_2d['y'], annotation_2d['y'] + annotation_2d['h'], annotation_2d['y'] + annotation_2d['h'], annotation_2d['y'], annotation_2d['y']]
    ax.plot(xs, ys, linewidth=1/5)
    ax.text(xs[0], ys[0], annotation_2d['id'], fontsize=2, color='r')

def draw_bev_box(ax, annotation_3d):
    x, y, width, height, angle = annotation_3d['x'], annotation_3d['y'], annotation_3d['l'], annotation_3d['h'], annotation_3d['alpha'],
    # print(angle*180/math.pi)
    anglePi = angle
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)
    x1 = x - 0.5 * width
    y1 = y - 0.5 * height
    x0 = x + 0.5 * width
    y0 = y1
    x2 = x1
    y2 = y + 0.5 * height
    x3 = x0
    y3 = y2
    x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
    y0n = (x0 - x) * sinA + (y0 - y) * cosA + y
    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y
    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y
    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y
    line1 = [(x0n, y0n), (x1n, y1n)]
    line2 = [(x1n, y1n), (x2n, y2n)]
    line3 = [(x2n, y2n), (x3n, y3n)]
    line4 = [(x0n, y0n), (x3n, y3n)]
    (line1_xs, line1_ys) = zip(*line1)
    (line2_xs, line2_ys) = zip(*line2)
    (line3_xs, line3_ys) = zip(*line3)
    (line4_xs, line4_ys) = zip(*line4)
    # print(line1_xs)
    # print(line1_ys)
    ax.add_line(Line2D(line1_xs, line1_ys, linewidth=0.3, color='g'))
    ax.add_line(Line2D(line2_xs, line2_ys, linewidth=0.3, color='g'))
    ax.add_line(Line2D(line3_xs, line3_ys, linewidth=0.3, color='g'))
    ax.add_line(Line2D(line4_xs, line4_ys, linewidth=0.3, color='g'))
    ax.text(x, y, annotation_3d['id'], fontsize=0.5, color='r')
    print( annotation_3d['id'],)

def draw_bev_box_tran_camera(ax, annotation_3d, RT_Matrix):
    x_l, y_l, z_l = annotation_3d['x'], annotation_3d['y'], annotation_3d['z']
    xyz = np.array([x_l, y_l, z_l, 1]).T
    xyz_camera = np.dot(RT_Matrix, xyz)
    x_l, y_l, z_l = xyz_camera[0], xyz_camera[1], xyz_camera[2]
    # print(RT_Matrix[:, 0:3])
    angle = annotation_3d['alpha']
    angle_Matrix = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    angle_Matrix_camera = np.dot(RT_Matrix[:, 0:3], angle_Matrix)
    r3 = R.from_matrix(angle_Matrix_camera)
    euler_1 = r3.as_euler('zyx', degrees=True)
    # print(euler_1)
    #
    # r4 = R.from_matrix(angle_Matrix)
    # euler_2 = r4.as_euler('zyx', degrees=True)
    # print(euler_2)

    x, y = x_l, z_l
    width, height = annotation_3d['l'], annotation_3d['h']
    angle = (euler_1[0]/180)*math.pi
    # print(angle)
    anglePi = angle
    # print(anglePi*180/math.pi)
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)
    x1 = x - 0.5 * width
    y1 = y - 0.5 * height
    x0 = x + 0.5 * width
    y0 = y1
    x2 = x1
    y2 = y + 0.5 * height
    x3 = x0
    y3 = y2
    x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
    y0n = (x0 - x) * sinA + (y0 - y) * cosA + y
    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y
    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y
    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y
    # print([(x0n, y0n), (x1n, y1n)], [(x1n, y1n), (x2n, y2n)], [(x2n, y2n), (x3n, y3n)], [(x0n, y0n), (x3n, y3n)])
    line1 = [(x0n, y0n), (x1n, y1n)]
    line2 = [(x1n, y1n), (x2n, y2n)]
    line3 = [(x2n, y2n), (x3n, y3n)]
    line4 = [(x0n, y0n), (x3n, y3n)]
    (line1_xs, line1_ys) = zip(*line1)
    (line2_xs, line2_ys) = zip(*line2)
    (line3_xs, line3_ys) = zip(*line3)
    (line4_xs, line4_ys) = zip(*line4)
    # print(line1_xs)
    # print(line1_ys)
    ax.add_line(Line2D(line1_xs, line1_ys, linewidth=0.3, color='g'))
    ax.add_line(Line2D(line2_xs, line2_ys, linewidth=0.3, color='g'))
    ax.add_line(Line2D(line3_xs, line3_ys, linewidth=0.3, color='g'))
    ax.add_line(Line2D(line4_xs, line4_ys, linewidth=0.3, color='g'))
    ax.text(x, y, annotation_3d['id'], fontsize=0.5, color='r')
    # print(annotation_3d['id'])

def compute_box_corners_3d(x,y,z,w,l,h,alpha):
    """Computes the 3D bounding box corner positions from an ObjectLabel

    :param object_label: ObjectLabel to compute corners from
    :return: a numpy array of 3D corners if the box is in front of the camera,
             an empty array otherwise
    """

    # Compute rotational matrix
    rot = np.array([[+np.cos(alpha), 0, +np.sin(alpha)],
                    [0, 1, 0],
                    [-np.sin(alpha), 0, +np.cos(alpha)]])
    # 3D BB corners
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z

    return corners_3d

def project_to_image(point_cloud, p):
    """ Projects a 3D point cloud to 2D points for plotting

    :param point_cloud: 3D point cloud (3, N)
    :param p: Camera matrix (3, 4)

    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """

    pts_2d = np.dot(p, point_cloud)

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d

def project_box3d_to_image(corners_3d, p):
    """Computes the 3D bounding box projected onto
    image space.

    Keyword arguments:
    obj -- object file to draw bounding box
    p -- transform matrix

    Returns:
        corners : numpy array of corner points projected
        onto image space.
        face_idx: numpy array of 3D bounding box face
    """
    # index for 3d bounding box face
    # it is converted to 4x4 matrix
    face_idx = np.array([0, 1, 5, 4,  # front face
                         1, 2, 6, 5,  # left face
                         2, 3, 7, 6,  # back face
                         3, 0, 4, 7]).reshape((4, 4))  # right face
    return project_to_image(corners_3d, p), face_idx

def draw_3d_box_img(ax, annotation_3d, RT_Matrix, IntrinsicMatrix):
    x_l, y_l, z_l = annotation_3d['x'], annotation_3d['y'], annotation_3d['z']
    xyz = np.array([x_l, y_l, z_l, 1]).T
    xyz_camera = np.dot(RT_Matrix, xyz)
    x_l, y_l, z_l = xyz_camera[0], xyz_camera[1], xyz_camera[2]
    # print(RT_Matrix[:, 0:3])
    angle = annotation_3d['alpha']
    angle_Matrix = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    angle_Matrix_camera = np.dot(RT_Matrix[:, 0:3], angle_Matrix)
    r3 = R.from_matrix(angle_Matrix_camera)
    euler_1 = r3.as_euler('zyx', degrees=True)
    angle = (euler_1[0] / 180) * math.pi
    w, l, h = annotation_3d['w'], annotation_3d['l'], annotation_3d['h']
    p = np.array(IntrinsicMatrix)

    corners3d = compute_box_corners_3d(x_l, y_l, z_l, w, l, h, angle)
    # print(corners3d)
    corners, face_idx = project_box3d_to_image(corners3d, p)
    if len(corners) > 0:
        for i in range(4):
            x = np.append(corners[0, face_idx[i, ]], corners[0, face_idx[i, 0]])
            y = np.append(corners[1, face_idx[i, ]],
                          corners[1, face_idx[i, 0]])

            ax.plot(x, y, linewidth=3,
                    color='g')

# def read_one_frame(camera_label_dir, camera_image_dir, lidar_label_dir, lidar_point_dir, radar_label_dir, radar_point_dir):
#     assert 'json' in camera_label_dir
#     with open(camera_label_dir) as f_camera_label:
#         camera_label = json.load(f_camera_label)
#     # print(camera_label)
#
#     assert 'png' in camera_image_dir
#     camera_image = cv.imread(camera_image_dir)
#     # cv.imshow('r', camera_image)
#     # cv.waitKey()
#
#     assert 'json' in lidar_label_dir
#     with open(lidar_label_dir) as f_lidar_label:
#         lidar_label = json.load(f_lidar_label)
#     # print(lidar_label)
#
#     assert 'pcd' in lidar_point_dir
#     lidar_points = load_VelodyneLidarPcd(lidar_point_dir)
#
#     # print(lidar_points)
#
#     assert 'json' in radar_label_dir
#     with open(radar_label_dir) as f_radar_label:
#         radar_label = json.load(f_radar_label)
#     # print(radar_label)
#
#     assert 'pcd' in radar_point_dir
#     radar_points = load_TIRadarPcd(radar_point_dir)
#
#     # print(radar_points)
#     fig = plt.figure(figsize=(20, 8))
#
#     VelodyneLidar_to_LeopardCamera1_TransformMatrix = np.array(lidar_label['VelodyneLidar_to_LeopardCamera1_TransformMatrix'])
#     IntrinsicMatrix = np.matrix(camera_label['IntrinsicMatrix'])
#     # lidar_points_in_img = np.dot(VelodyneLidar_to_LeopardCamera1_TransformMatrix, np.concatenate((lidar_points[:, 0:3], np.ones_like(lidar_points)[:, 0:1]), axis=1).T)
#     # lidar_points_in_camera = np.array(np.dot(IntrinsicMatrix.I, lidar_points_in_img))
#
#     # lidar_points_in_img = lidar_points_in_img/lidar_points_in_img[2, :]
#     # lidar_points_bool_in_img_1 = np.logical_and((lidar_points_in_img[0, :] > 0), (lidar_points_in_img[1, :] > 0))
#     # lidar_points_bool_in_img_2 = np.logical_and((lidar_points_in_img[0, :] < camera_image.shape[1]), (lidar_points_in_img[1, :] < camera_image.shape[0]))
#     # lidar_points_bool_in_img = np.logical_and(lidar_points_bool_in_img_1, lidar_points_bool_in_img_2)
#     # lidar_points_bool_in_front = (lidar_points_in_camera[2, :] > 0)
#     # lidar_points_bool_total = np.logical_and(lidar_points_bool_in_img, lidar_points_bool_in_front)
#     # lidar_points_in_img = lidar_points_in_img[:, lidar_points_bool_total]
#     # lidar_points_in_camera = lidar_points_in_camera[:, lidar_points_bool_total]
#
#     # TIRadar_to_LeopardCamera1_TransformMatrix = np.array(radar_label['TIRadar_to_LeopardCamera1_TransformMatrix'])
#     # IntrinsicMatrix = np.matrix(camera_label['IntrinsicMatrix'])
#     # radar_points_in_img = np.dot(TIRadar_to_LeopardCamera1_TransformMatrix,
#     #                              np.concatenate((radar_points[:, 0:3], np.ones_like(radar_points)[:, 0:1]), axis=1).T)
#     # radar_points_in_camera = np.array(np.dot(IntrinsicMatrix.I, radar_points_in_img))
#     #
#     # radar_points_in_img = radar_points_in_img / radar_points_in_img[2, :]
#     # radar_points_bool_in_img_1 = np.logical_and((radar_points_in_img[0, :] > 0), (radar_points_in_img[1, :] > 0))
#     # radar_points_bool_in_img_2 = np.logical_and((radar_points_in_img[0, :] < camera_image.shape[1]),
#     #                                             (radar_points_in_img[1, :] < camera_image.shape[0]))
#     # radar_points_bool_in_img = np.logical_and(radar_points_bool_in_img_1, radar_points_bool_in_img_2)
#     # radar_points_bool_in_front = (radar_points_in_camera[2, :] > 0)
#     # radar_points_bool_total = np.logical_and(radar_points_bool_in_img, radar_points_bool_in_front)
#     # radar_points_in_img = radar_points_in_img[:, radar_points_bool_total]
#     # radar_points_in_camera = radar_points_in_camera[:, radar_points_bool_total]
#
#     annotations_3d = lidar_label['annotation']
#     RT_Matrix = np.array(np.dot(IntrinsicMatrix.I, VelodyneLidar_to_LeopardCamera1_TransformMatrix))
#     ax12 = fig.add_subplot(1, 1, 1)
#     camera_image = camera_image[:,:,::-1]
#     ax12.imshow(camera_image)
#     for annotation_3d in annotations_3d:
#         draw_3d_box_img(ax12, annotation_3d, RT_Matrix, IntrinsicMatrix)
#     plt.show()
#     # plt.savefig('3DBox.png', bbox_inches='tight', pad_inches=-0.1)
# #     , dpi=500


def draw_3D_Box(camera_label_dir, camera_image_dir, lidar_label_dir, folder):
    assert 'json' in camera_label_dir
    with open(camera_label_dir) as f_camera_label:
        camera_label = json.load(f_camera_label)
    # print(camera_label)

    assert 'png' in camera_image_dir
    camera_image = cv.imread(camera_image_dir)
    # cv.imshow('r', camera_image)
    # cv.waitKey()

    assert 'json' in lidar_label_dir
    with open(lidar_label_dir) as f_lidar_label:
        lidar_label = json.load(f_lidar_label)
    # print(lidar_label)
    fig = plt.figure(figsize=(20, 8))

    VelodyneLidar_to_LeopardCamera1_TransformMatrix = np.array(lidar_label['VelodyneLidar_to_LeopardCamera1_TransformMatrix'])
    IntrinsicMatrix = np.matrix(camera_label['IntrinsicMatrix'])

    annotations_3d = lidar_label['annotation']
    RT_Matrix = np.array(np.dot(IntrinsicMatrix.I, VelodyneLidar_to_LeopardCamera1_TransformMatrix))
    ax12 = fig.add_subplot(1, 1, 1)
    camera_image = camera_image[:,:,::-1]
    ax12.imshow(camera_image)
    for annotation_3d in annotations_3d:
        draw_3d_box_img(ax12, annotation_3d, RT_Matrix, IntrinsicMatrix)
    # plt.title(camera_image_dir)
    # plt.show()
    # save_name = camera_image_dir.split('/')[-1]
    save_name = folder + '.png'
    plt.savefig(save_name, bbox_inches='tight', pad_inches=-0.1)
#     , dpi=500


# base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled/20211025_1_group0012_frame0045_labeled'
# camera_label_dir = os.path.join(base_path, 'LeopardCamera1', '1635145269.477.json')
# camera_image_dir = os.path.join(base_path, 'LeopardCamera1', '1635145269.477.png')
# lidar_label_dir = os.path.join(base_path, 'VelodyneLidar', '1635145269.014.json')
#
#
#
# draw_3D_Box(camera_label_dir, camera_image_dir, lidar_label_dir)

# base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/tmp2'
# base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train'
# base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'
# base_path = '/media/ourDataset/v1.0_label/20211027_1_group0010_148frames_30labeled'
base_path = '/mnt/ourDataset_v1/ourDataset_v1_label/20211025_1_group0012_185frames_37labeled'
folders = os.listdir(base_path)
folders = sorted(folders)
# frame_num = -1
for folder in folders:
    if 'labeled' not in folder:
        continue
    camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
    for file in os.listdir(camera_path):
        if file[-3:] == 'png':
            img_path = os.path.join(camera_path, file)
        if file[-4:] == 'json':
            img_json_path = os.path.join(camera_path, file)
    lidar_path = os.path.join(base_path, folder, 'VelodyneLidar')
    for file in os.listdir(lidar_path):
        if file[-3:] == 'pcd':
            pcd_lidar = os.path.join(lidar_path, file)
        if file[-4:] == 'json':
            calib_lidar = os.path.join(lidar_path, file)

    # base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled/20211025_1_group0012_frame0045_labeled'
    # camera_label_dir = os.path.join(base_path, 'LeopardCamera1', '1635145269.477.json')
    # camera_image_dir = os.path.join(base_path, 'LeopardCamera1', '1635145269.477.png')
    # lidar_label_dir = os.path.join(base_path, 'VelodyneLidar', '1635145269.014.json')

    draw_3D_Box(img_json_path, img_path, calib_lidar, folder)
    print()

