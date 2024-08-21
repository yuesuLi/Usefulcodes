import os
import cv2 as cv
import random
import pandas as pd
import json
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters



def load_TIRadarHeatmap(heatmap_path):
    '''
    read TI radar heatmap
    :param heatmap_path: str - TI radar heatmap path
    :return: dict(np.array)
    '''
    data = np.fromfile(heatmap_path, dtype='float32')
    # data = data.reshape((232, 4*257)).T
    # data = data.reshape((4, 257, 232))
    data = data.reshape((4*257, 232), order='F')
    data = data.reshape((4, 257, 232))
    res = {
        "heatmap_static": data[0, :, :],
        "heatmap_dynamic": data[1, :, :],
        "x_bins": data[2, :, :],
        "y_bins": data[3, :, :],
    }
    return res


def get_data(mmxx, mmyy, mmzz):

    df = pd.DataFrame(data=[v for v in zip(mmxx, mmyy, mmzz)], columns=['x', 'y', 'z'])
    return df

def plot_heatmap(heatmap):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #  cmap=plt.get_cmap('rainbow')
    # heatmap_static, heatmap_dynamic
    surf = ax.plot_surface(heatmap['x_bins'], heatmap['y_bins'], heatmap['heatmap_dynamic'])
    ax.view_init(elev=90, azim=-90)
    # plt.savefig('./heatmap.png')
    plt.show()


def imshow_heatmap(heatmap, frame_num, GTs):
    GTs = np.array(GTs)
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, interpolation='gaussian')
    # ax.invert_yaxis()
    X_bins = GTs[:, 0]
    Y_bins = GTs[:, 1]
    angel = np.arctan(Y_bins/X_bins)
    distance = np.sqrt(X_bins**2 + Y_bins**2)

    height = heatmap.shape[0]
    width = heatmap.shape[1]
    y = [(height - tmp_y / 75 * height) for tmp_y in distance]
    x = [(width/2 - (tmp_x / 3.14 * width)) for tmp_x in angel]

    plt.scatter(np.array(x)[:].astype(np.int32), np.array(y)[:].astype(np.int32), s=100, c='k', marker='*')
    # plt.scatter(np.array(GTs)[:, 0].astype(np.int32), np.array(GTs)[:, 0].astype(np.int32), s=10, c='k', marker='*')
    # plt.axis('off')
    plt.show()
    # save_name = '/home/zhangq/Desktop/zhangq/UsefulCode/imgs/{}.png'.format(frame_num)
    # plt.savefig(save_name, bbox_inches='tight', pad_inches=-0.1)


def OurLabel2RADLabel_raw(lidar_json_data):

    x, y, l, w, alpha, classes = [], [], [], [], [], []
    cart_boxes = []
    has_label = False
    for idx in range(len(lidar_json_data['annotation'])):
        if lidar_json_data['annotation'][idx]['x'] >= 50 or lidar_json_data['annotation'][idx]['x'] <= 0:
            continue
        x.append(lidar_json_data['annotation'][idx]['x'])
        y.append(lidar_json_data['annotation'][idx]['y'])
        l.append(lidar_json_data['annotation'][idx]['l'])
        w.append(lidar_json_data['annotation'][idx]['w'])
        # xywh = [lidar_json_data['annotation'][idx]['y'], lidar_json_data['annotation'][idx]['x'], \
        #         lidar_json_data['annotation'][idx]['w'], lidar_json_data['annotation'][idx]['l']]
        alpha.append(lidar_json_data['annotation'][idx]['alpha'])
        if lidar_json_data['annotation'][idx]['class'] is 'pedestrian':
            lidar_json_data['annotation'][idx]['class'] = 'person'
        classes.append(lidar_json_data['annotation'][idx]['class'])
        has_label = True

    # width = 256
    # height = 256
    # # x = [tmp_x / 75 * width for tmp_x in x]
    # # y = [(-tmp_y + 37.5) / 75 * height for tmp_y in y]
    # # l = [tmp_l / 75 * width for tmp_l in l]
    # # w = [tmp_w / 75 * height for tmp_w in w]
    #
    # if has_label:
    #     for i in range(len(x)):
    #         xywh = [y[i], x[i], w[i], l[i]]
    #         cart_boxes.append(xywh)
    #
    #     # cart_boxes = np.array(cart_boxes)
    #     X_bins = np.array(cart_boxes)[:, 0]
    #     Y_bins = np.array(cart_boxes)[:, 1]
    #     angel = np.arctan(Y_bins / X_bins)
    #     distance = np.sqrt(X_bins ** 2 + Y_bins ** 2)
    #     x = [(height - tmp_x / 75 * height) for tmp_x in distance]
    #     y = [(width / 2 - (tmp_y / 3.14 * width)) for tmp_y in angel]

    for i in range(len(x)):
        xywh = [y[i], x[i], w[i], l[i]]
        cart_boxes.append(xywh)

    gt_instances = {'classes': classes, 'cart_boxes': cart_boxes}

    return gt_instances

def OurLabel2RADLabel(lidar_json_data, TI_json_data):

    x, y, z, l, w, alpha, classes = [], [], [], [], [], [], []
    cart_boxes = []
    for idx in range(len(lidar_json_data['annotation'])):
        if lidar_json_data['annotation'][idx]['x'] >= 50 or lidar_json_data['annotation'][idx]['x'] <= 0 or \
                np.sqrt(lidar_json_data['annotation'][idx]['x']**2 + lidar_json_data['annotation'][idx]['y']**2) >= 75:
            continue
        x.append(lidar_json_data['annotation'][idx]['x'])
        y.append(lidar_json_data['annotation'][idx]['y'])
        z.append(lidar_json_data['annotation'][idx]['z'])
        l.append(lidar_json_data['annotation'][idx]['l'])
        w.append(lidar_json_data['annotation'][idx]['w'])
        # xywh = [lidar_json_data['annotation'][idx]['y'], lidar_json_data['annotation'][idx]['x'], \
        #         lidar_json_data['annotation'][idx]['w'], lidar_json_data['annotation'][idx]['l']]
        alpha.append(lidar_json_data['annotation'][idx]['alpha'])
        if lidar_json_data['annotation'][idx]['class'] == 'pedestrian':
            lidar_json_data['annotation'][idx]['class'] = 'person'
        classes.append(lidar_json_data['annotation'][idx]['class'])

    # width = 256
    # height = 256
    # x = [tmp_x / 75 * width for tmp_x in x]
    # y = [(-tmp_y + 37.5) / 75 * height for tmp_y in y]
    # l = [tmp_l / 75 * width for tmp_l in l]
    # w = [tmp_w / 75 * height for tmp_w in w]

    x = np.array(x).reshape((1, len(x)))
    y = np.array(y).reshape((1, len(y)))
    z = np.array(z).reshape((1, len(z)))
    xyz = np.concatenate((y,-x,z))
    xyz1 = np.concatenate((xyz, np.ones([1, xyz.shape[1]])))  # 给xyz在后面叠加了一行1 (4,N)

    P_TI = np.array(TI_json_data["TIRadar_to_LeopardCamera1_TransformMatrix"])
    P_Lidar = np.array(lidar_json_data["VelodyneLidar_to_LeopardCamera1_TransformMatrix"])
    R_TI = P_TI[:, 0:3]
    T_TI = P_TI[:, 3].reshape((3,1))

    R_Lidar = P_Lidar[:, 0:3]
    T_Lidar = P_Lidar[:, 3].reshape((3, 1))

    R = np.dot(np.linalg.inv(R_TI), R_Lidar)
    T = np.dot(np.linalg.inv(R_TI), (T_Lidar - T_TI))

    P = np.concatenate((R, T), axis=1)
    xyz1_TI = np.dot(P, xyz1)  # (3,N)  TI坐标系下的xyz坐标


    for i in range(len(x)):
        xywh = [xyz1_TI[0, i], xyz1_TI[1, i], w[i], l[i]]
        cart_boxes.append(xywh)

    gt_instances = {'classes': classes, 'cart_boxes': cart_boxes}

    return gt_instances


def test():


    # base_path = '/media/ourDataset/v1.0_label/20211027_1_group0021_123frames_25labeled'
    # base_path = '/home/zhangq/Desktop/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'
    # base_path = '/home/zhangq/Desktop/ourDataset/v1.0_label/20211027_2_group0079_109frames_22labeled'
    # base_path = '/media/ourDataset/v1.0_label/20211027_1_group0026_144frames_29labeled'
    # base_path = '/media/ourDataset/v1.0_label/20211027_1_group0028_134frames_27labeled'
    # base_path = '/home/zhangq/Desktop/zhangq/heatmap_check/data_test/heatmap_example'
    # base_path = '/media/ourDataset/v1.0_label/20211027_1_group0010_148frames_30labeled'
    # base_path = './20211027_2_group0051_72frames_15labeled'
    # base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train'
    base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train'
    save_path = './save/test/'

    folders = os.listdir(base_path)
    folders = sorted(folders)
    frame_num = -1
    for folder in folders:
        frame_num += 1
        # if frame_num > 1:
        #     break
        print('frame_num:', frame_num)
        print('scene:', folder)

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
                TI_json_path = os.path.join(ti_path, file)
            if file[-11:] == 'heatmap.bin':
                TI_heatmap_path = os.path.join(ti_path, file)

        img = cv.imread(img_path)
        # width, height = img.shape[1], img.shape[0]

        with open(Lidar_json_path) as f:
            lidar_json_data = json.load(f)

        with open(TI_json_path) as f:
            TI_json_data = json.load(f)

        gt_instances = OurLabel2RADLabel_raw(lidar_json_data)
        print('gt_instances', gt_instances)
        # print('cart_boxes', np.array(gt_instances['cart_boxes']).shape)

        TI_heatmap_data = load_TIRadarHeatmap(TI_heatmap_path)
        # plot_heatmap(TI_heatmap_data)

        TI_heatmap_static = TI_heatmap_data['heatmap_static']
        TI_heatmap_dynamic = TI_heatmap_data['heatmap_dynamic']
        # X_bins = TI_heatmap_data['x_bins']
        # Y_bins = TI_heatmap_data['y_bins']
        # angel = np.arctan2(X_bins, Y_bins) / (np.pi/2)
        # distance = np.sqrt(X_bins**2 + Y_bins**2)
        # print('X_bins:', X_bins.shape)
        # print('Y_bins:', Y_bins.shape)


        # TEST = X_bins[128:, :]
        TI_heatmap_dynamic = np.vstack((TI_heatmap_dynamic[128:, :], TI_heatmap_dynamic[0:128, :]))
        TI_heatmap_dynamic = TI_heatmap_dynamic.T
        # TI_heatmap_dynamic = TI_heatmap_dynamic[0:128, 128-64:128+64]
        TI_heatmap_dynamic = TI_heatmap_dynamic[::-1, :]
        TI_heatmap_dynamic[0:TI_heatmap_dynamic.shape[0]//3, :] = np.mean(TI_heatmap_dynamic)
        TI_heatmap_dynamic = TI_heatmap_dynamic/np.max(TI_heatmap_dynamic)


        # TI_heatmap_static = np.vstack((TI_heatmap_static[128:, :], TI_heatmap_static[0:128, :]))
        # TI_heatmap_static = TI_heatmap_static.T
        # # TI_heatmap_static = TI_heatmap_static[0:128, 128 - 64:128 + 64]
        # TI_heatmap_static = TI_heatmap_static[::-1, :]
        # TI_heatmap_static = TI_heatmap_static / np.max(TI_heatmap_static)

        # print('TI_heatmap_dynamic:', TI_heatmap_dynamic.shape)
        # print('TI_heatmap_static:', TI_heatmap_static.shape)

        # plt.pcolor(X_bins, Y_bins, TI_heatmap_dynamic)
        img = cv.resize(img, (1080, 720))
        cv.imshow('test-img', img)
        if cv.waitKey(0) & 0xFF == 27:
            break

        imshow_heatmap(TI_heatmap_dynamic, frame_num, gt_instances['cart_boxes'])
        # plt.show()

        print(' ')
        # neighborhood_size = 5
        # threshold = 0.4
        # TI_heatmap_dynamic[TI_heatmap_dynamic < threshold] = 0.
        # dynamic_max = filters.maximum_filter(TI_heatmap_dynamic, neighborhood_size)
        # maxima = (TI_heatmap_dynamic == dynamic_max)
        # maxima[TI_heatmap_dynamic == 0.] = False
        # print('where:', np.where(maxima==True))
        #
        # labeled, num_objects = ndimage.label(maxima)
        #
        # slices = ndimage.find_objects(labeled)
        # x, y = [], []
        # for dx, dy in slices:
        #     x_center = (dx.start + dx.stop - 1) / 2
        #     x.append(x_center)
        #     y_center = (dy.start + dy.stop - 1) / 2
        #     y.append(y_center)
        # # print('slices:', slices)
        # x = np.array(x).astype(np.int32)
        # y = np.array(y).astype(np.int32)
        # print('TI_heatmap_dynamic_peak:', TI_heatmap_dynamic[x ,y])
        # print('x:', x, 'y:', y)


        # print(Y_bins.shape)
        # ax = plt.subplot(111)
        # plt.xlim(np.min(X_bins), np.max(X_bins))
        # plt.ylim(np.min(Y_bins), np.max(Y_bins))
        # TI_heatmap_dynamic = TI_heatmap_dynamic / np.max(TI_heatmap_dynamic)
        # # ax.fill_between()
        #
        # plt.show()
        # print('X_bins:', X_bins)
        # print('Y_bins:', Y_bins)
        # print('angel:', angel)
        # print('distance:', distance)
        # plot_heatmap(TI_heatmap_data, TI_heatmap_static)



    print('test done.')


if __name__ == "__main__":
    test()


