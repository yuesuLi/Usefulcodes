# Semantic-guided depth completion
import numpy as np
import cv2
import mayavi.mlab
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
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
    print(f'pcd points: {points.shape}')

    if pts_view:
        ptsview(points)
    return points


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


def pts2rbev(lpts):

    # Remove LiDAR ground points
    from ransac import my_ransac_v3
    indices, model = my_ransac_v3(lpts[:, :3], distance_threshold=0.7)
    lpts = np.delete(lpts, indices, axis=0)
    # print(l.shape)
    # ptsview(d)
    # raise NotImplemented
    # from ground_removal import Processor
    # process = Processor(n_segments=140, n_bins=180, line_search_angle=0.3, max_dist_to_line=2.15,
    #                     sensor_height=2.53, max_start_height=-2.5, long_threshold=8)
    # vel_non_ground = process(lpts)
    # ptsview(vel_non_ground)
    #
    # raise NotImplemented
    # LiDAR points to radar coordinate
    VelodyneLidar_to_LeopardCamera1_TransformMatrix = np.array(
        [
            [1878.13694567694, -1900.51229133004, -3.76435830142225, -1450.90535286182],
            [962.699273002558, 8.21051460132727, -1996.69155762544, -1289.34979520961],
            [0.998966844122018, 0.0410508475655525, -0.0194954420069267, -0.515288952567310]
        ]
    )
    TIRadar_to_LeopardCamera1_TransformMatrix = np.array(
        [
            [2019.61363529422, 1745.88166828988, -111.433796801951, -419.938881768377],
            [26.0193673714885, 870.796981112031, -2038.30078479358, -120.997110351106],
            [0.0244308479903333, 0.997614077965117, -0.0645700016438225, -0.00641535834610336]
        ]
    )
    LeopardCamera1_IntrinsicMatrix = np.array(
        [
            [1976.27129878769, 0, 1798.25228491297],
            [0, 1977.80114435384, 1000.96808764067],
            [0, 0, 1]
        ]
    )
    VelodyneLidar_to_TIRadar_TransformMatrix = np.matmul(
        np.linalg.inv(
            np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                                 TIRadar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
        ),
        np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                             VelodyneLidar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
    )

    pts = pointcloud_transform(lpts, VelodyneLidar_to_TIRadar_TransformMatrix)

    # ptsview(lpts)
    # ptsview(pts)

    side_range = (-75, 75)
    fwd_range = (0, 75)
    height_range = (-30, 50)  # height_range should be modified manually
    y, x, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # print(np.min(x), np.max(x)) # -199.20, 197.77
    # print(np.min(y), np.max(y)) # -185.53, 196.74
    # print(np.min(z), np.max(z)) # -5.02, 39.75
    f_filter = np.logical_and(x > fwd_range[0], x < fwd_range[1])
    s_filter = np.logical_and(y > side_range[0], y < side_range[1])
    h_filter = np.logical_and(z > height_range[0], z < height_range[1])
    filter = np.logical_and(f_filter, s_filter)
    filter = np.logical_and(filter, h_filter)  # height filter
    indices = np.argwhere(filter).flatten()
    x, y, z = x[indices], y[indices], z[indices]

    res = 0.25
    x_img = (-y / res).astype(np.int32)
    y_img = (-x / res).astype(np.int32)
    x_img = x_img - int(np.floor(side_range[0]) / res)
    y_img = y_img + int(np.floor(fwd_range[1]) / res)

    # pixel_value = np.clip(a=z, a_max=height_range[1], a_min=height_range[0])
    pixel_value = 255  # z

    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)

    # pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    # pixel_value = (pixel_value - np.min(pixel_value)) / (np.max(pixel_value) - np.min(pixel_value)) * 255
    x_max = int((side_range[1] - side_range[0]) / res) + 1
    y_max = int((fwd_range[1] - fwd_range[0]) / res) + 1

    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    # im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
    # print(im.shape)
    # plt.imshow(im, cmap='jet')
    # plt.show()
    return im[:, ::-1]


def pltRadLid(lidar):
    def norm_image(image):
        image = image.copy()
        image -= np.max(np.min(image), 0)
        image /= np.max(image)
        image *= 255
        return np.uint8(image)

    masks = norm_image(np.float32(lidar)).astype(np.uint8)
    heatmap = cv2.applyColorMap(masks, cv2.COLORMAP_HOT)
    heatmap = np.float32(heatmap)

    # cam = 0.2 * heatmap + 0.8 * np.float32(radar)
    # Image.fromarray(np.uint8(cam)).show()
    return heatmap

if __name__ == '__main__':
    # base_path = './Dataset'
    base_path = '/home/zhangq/Desktop/ourDataset/v1.0_label/20211031_1_group0020_104frames_21labeled'
    base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'
    folders = os.listdir(base_path)
    folders = sorted(folders)
    for folder in folders:
        camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
        for file in os.listdir(camera_path):
            if file[-3:] == 'png':
                img_path = os.path.join(camera_path, file)
        lidar_path = os.path.join(base_path, folder, 'VelodyneLidar')
        for file in os.listdir(lidar_path):
            if file[-3:] == 'pcd':
                pcd_lidar = os.path.join(lidar_path, file)
            if file[-4:] == 'json':
                calib_lidar = os.path.join(lidar_path, file)

        # Bev image, where lidar points transformed into radar coordinate
        lid_im = pts2rbev(read_pcd(pcd_lidar))

        # alignment between 'lidar Bev' and 'radar RF'
        # cam = pltRadLid((colored_image[:, :, :3] * 255).astype(np.uint8), lid_im)
        cam = pltRadLid(lid_im)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        # cam = cv2.resize(cam, (1080, 720))
        cv2.imshow('test', cam)
        cv2.waitKey(0)
        cv2.destroyAllWindows()