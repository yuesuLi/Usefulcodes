import os
import re
import json
import cv2
import mayavi
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
                             pc_data[metadata['fields'][3]][:, None],
                             pc_data[metadata['fields'][4]][:, None]], axis=-1)
    # print(points.shape)

    if pts_view:
        ptsview(points)
    return points

def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        return json_data

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


    return radar_2d

# uv:(N,4)
def draw_points_to_img(uv, img, size=4, num_clusters=1, db=None, colors=None):
    points = np.array((uv[:, 0].astype(int), uv[:, 1].astype(int))).T  # (N, 2)
    if num_clusters == 0:
        return

    if db == None:
        for i in range(points.shape[0]):
            depth = uv[i, 2]
            color = cmap[int(3 * depth), :]
            cv2.circle(img, (points[i, 0], points[i, 1]), size, color=tuple(color), thickness=-1)
        return
    color_list = plt.cm.tab20(np.linspace(0, 1, num_clusters))*255
    for i in range(num_clusters):
        points_class = points[db.labels_ == i, :]
        color = color_list[i]
        for j in range(points_class.shape[0]):
            cv2.circle(img, (points_class[j, 0].astype(int), points_class[j, 1].astype(int)), size, color=tuple(color), thickness=-1)


def draw_points(points, order=1, num_clusters=1, db=None, colors=None, point_size=1):
    # plt.subplot(2, 2, order)
    if num_clusters == 0:
        return
    flag = 1

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

    # plt.axis('scaled')  # {equal, scaled}
    plt.xlabel('X')
    plt.ylabel('Y')



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

    return x, y, z, l, w, h, alpha

# base_path = '/media/ourDataset/v1.0_label/20211027_1_group0010_148frames_30labeled'
base_path = '/media/ourDataset/v1.0_label/20211031_1_group0001_240frames_44labeled'
# base_path = '/media/ourDataset/v1.0_label/20211027_1_group0028_134frames_27labeled'
save_path = './save/test/'


folders = os.listdir(base_path)
folders = sorted(folders)
frame_num = -1
for folder in folders:
    frame_num += 1
    print('frame_num:', frame_num)

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
    ti_path = os.path.join(base_path, folder, 'TIRadar')
    for file in os.listdir(ti_path):
        if file[-3:] == 'pcd':
            TI_pcd_path = os.path.join(ti_path, file)
        if file[-4:] == 'json':
            TI_radar_json_path = os.path.join(ti_path, file)

    img = cv2.imread(img_path)
    width, height = img.shape[1], img.shape[0]
    radar_img = np.zeros((height, width, 3), dtype=np.uint8)
    raw_img = cv2.imread(img_path)
    TI_img = raw_img.copy()


    lidar_pts = read_pcd(Lidar_pcd_path)
    print('lidar_pts', lidar_pts.shape)
    lidar_json_data = load_json(Lidar_json_path)

    x, y, z, l, w, h, alpha = get_lidar_annotation(Lidar_json_path)

    TI_radar_points = read_pcd(TI_pcd_path)
    mask = (TI_radar_points[:, 1] < 70) & (TI_radar_points[:, 1] > 5)
    TI_radar_points = TI_radar_points[mask, :]
    TI_value = np.zeros((TI_radar_points.shape[0], 1))
    print('TI_points: ', TI_radar_points.shape)

    for i in range(len(x)):
        flag = i + 1
        for j in range(TI_radar_points.shape[0]):
            if TI_radar_points[j][0] >= x[i] - l[i]/2 and TI_radar_points[j][0] <= x[i] + l[i]/2 and \
                TI_radar_points[j][1] >= y[i] - w[i] / 2 and TI_radar_points[j][1] <= y[i] + w[i] / 2:

                TI_value[j] = flag

    mask = TI_value[:, 0] != 0
    TI_value = TI_value[mask, :]
    print('TI_value', TI_value)



    TI_radar_points_xyv = np.concatenate((TI_radar_points[:, 0:2], TI_radar_points[:, 3:4]*0.5), axis=1)
    TI_radar_json_data = load_json(TI_radar_json_path)
    TI_TransformMatrix = np.array(TI_radar_json_data['TIRadar_to_LeopardCamera1_TransformMatrix'])


    plt.suptitle(str(frame_num) + '_BEV radar points')        # main title
    draw_points(TI_radar_points, 2)
    plt.title('TI_raw')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)     #
    # plt.savefig('./BEV_points.jpg')
    # BEV_savename = save_path + str(frame_num) + '_BEV_points.jpg'
    # plt.savefig(BEV_savename)
    # plt.clf()
    plt.show()

    # get_uv_from_points   return(N,4)  u,v,depth,velo
    TI_uv = get_uv_from_points(TI_radar_points, TI_TransformMatrix, width, height)

    # OCU_radar_value = np.zeros((width, height, 2), dtype=np.float32)
    # TI_radar_value = np.zeros((width, height, 2), dtype=np.floaclass TrackerTransformerConfig(BaseModel):

    TI_size = 5
    draw_points_to_img(TI_uv, TI_img, TI_size)
    # cv2.imshow('raw_img', cv2.resize(TI_img, (1080, 720)))
    # cv2.waitKey()


    # imgs1 = np.hstack([OCU_img, TI_img])
    # imgs2 = np.hstack([OCU_img_dbscan, TI_img_dbscan])
    # imgs = np.vstack([imgs1, imgs2])
    #
    #
    # points_savename = save_path + str(frame_num) + '_points_on_img.jpg'
    # cv2.imwrite(points_savename, imgs)
    # cv2.imwrite('points_on_img.jpg', imgs)

    print()


