# draw heatmap for every GT

import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt


def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)


    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap) # output replace masked_heatmap

    # plt.pcolor(heatmap)
    # plt.show()
    # np.set_printoptions(threshold=1e5)
    # print(heatmap.shape)
    return heatmap

def get_lidar_annotation(lidar_json_path):

    with open(lidar_json_path) as f:
        lidar_json_data = json.load(f)
    x, y, z, l, w, h, alpha = [], [], [], [], [], [], []
    rate = []
    # print("json_data:", type(lidar_json_data['annotation'][0]['x']))
    # for idx in range(len(lidar_json_data['annotation'])):
    #     if lidar_json_data['annotation'][idx]['x'] >= 75 or lidar_json_data['annotation'][idx]['x'] <= 0\
    #             or lidar_json_data['annotation'][idx]['y'] >= 75 or lidar_json_data['annotation'][idx]['y'] <= -75:
    #         continue
    #     x.append(lidar_json_data['annotation'][idx]['x'])
    #     y.append(lidar_json_data['annotation'][idx]['y'])
    #     l.append(lidar_json_data['annotation'][idx]['l'])
    #     w.append(lidar_json_data['annotation'][idx]['w'])
    #     alpha.append(lidar_json_data['annotation'][idx]['alpha'])
    #     rate.append((np.arctan2(lidar_json_data['annotation'][idx]['y'], lidar_json_data['annotation'][idx]['x'])) / np.pi)
    width = 543
    height = 429
    # rho = []
    # theta = []
    # x.append(53)
    # y.append(53)

    rho = np.sqrt(53**2+53**2)
    theta = np.arctan2(53, 53)
    r = 75/np.sin(theta)
    test_x = rho*429/r
    test_y = test_x/np.tan(theta) + 543/2
    print('', rho, r, test_x, test_y)

    x.append(test_x)
    y.append(test_y)
    l.append(10)
    w.append(5)
    alpha.append(1.57/2)
    # rate.append(0.5)



    # x = [(74 - tmp_x) / 74 * width for tmp_x in x]
    # y = [(tmp_y + 75) / 150 * height for tmp_y in y]
    l = [tmp_l / 74 * width for tmp_l in l]
    w = [tmp_w / 150 * height for tmp_w in w]

    # my_y = []
    # x = [(74 - tmp_x) / 74 * height for tmp_x in x]
    # for tmp_y, tmp_rate in zip(y, rate):
    #     # print('tmp_y:', tmp_y)
    #     # print('tmp_rate:', tmp_rate)
    #     tmp = tmp_y * tmp_rate * width
    #     my_y.append(tmp)
    # y = my_y
    # print('y:', y)
    # y = [tmp_y  * width for tmp_y, tmp_rate in zip(y, rate)]
    # l = [tmp_l / 74 * height for tmp_l in l]
    # w = [tmp_w / 150 * width for tmp_w in w]



    # print("x:", x)
    # print("y:", y)
    # print("l:", l)
    # print("w:", w)
    # print("alpha:", alpha)
    return x, y, l, w, alpha

def get_RGB_annotation(RGB_json_path):

    with open(RGB_json_path) as f:
        RGB_json_data = json.load(f)
    x, y, w, h = [], [], [], []
    # print("json_data:", type(lidar_json_data['annotation'][0]['x']))
    for idx in range(len(RGB_json_data['annotation'])):
        if RGB_json_data['annotation'][idx]['x'] >= 3517 or RGB_json_data['annotation'][idx]['x'] <= 0\
                or RGB_json_data['annotation'][idx]['y'] >= 1700 or RGB_json_data['annotation'][idx]['y'] <= 0:
            continue
        if RGB_json_data['annotation'][idx]['class'] != 'car':
            continue
        x.append(RGB_json_data['annotation'][idx]['x'])
        y.append(RGB_json_data['annotation'][idx]['y'])
        w.append(RGB_json_data['annotation'][idx]['w'])
        h.append(RGB_json_data['annotation'][idx]['h'])

    return x, y, w, h


# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners

def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    cv2.line(img, (int(corners_int[0, 0]), int(corners_int[0, 1])), (int(corners_int[3, 0]), int(corners_int[3, 1])), (0, 0, 255), 2)

def drawBox(img, x, y, w, h, color, yaw=0):
    left_top = np.array([x, y]).astype(int)
    right_top = np.array([x+w, y]).astype(int)
    left_bottom = np.array([x, y + h]).astype(int)
    right_bottom = np.array([x + w, y + h]).astype(int)

    cv2.rectangle(img, left_top, right_bottom, color=color, thickness=5)

    # cv2.line(img, left_top, right_top, color, thickness=2)
    # cv2.line(img, left_top, left_bottom, color, thickness=2)
    # cv2.line(img, right_top, right_bottom, color, thickness=2)
    # cv2.line(img, left_bottom, right_bottom, color, thickness=2)


def main():

    # base_path = '/media/personal_data/zhangq/TransT-main/dataset/mydata/20211027_1_group0021_frame0000_labeled'
    # # base_path = '/home/zhangq/Desktop/zhangq/heatmap_check/data_test/heatmap_example/20211025_1_group0012_frame0045_labeled'
    # # lidar_json_path = '/media/personal_data/zhangq/TransT-main/1635145120.294.json'
    # # lidar_json_path = os.path.join(base_path, 'VelodyneLidar', '1635319097.416.json')
    # img_path = os.path.join(base_path, 'LeopardCamera1', '1635319097.928.png')
    # img_json_path = os.path.join(base_path, 'LeopardCamera1', '1635319097.928.json')
    # # dynamic_heatmap_path = os.path.join(base_path, 'TIRadar', 'dynamic_heatmap_1635319097.410.png')
    # # static_heatmap_path = os.path.join(base_path, 'TIRadar', 'static_heatmap_1635319097.410.png')
    # # x, y, l, w, alpha = get_lidar_annotation(lidar_json_path)
    #
    # base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled/20211025_1_group0012_frame0045_labeled'
    # base_path = '/home/zhangq/Desktop/ourDataset/v1.0_label/20211025_1_group0031_149frames_30labeled/20211025_1_group0031_frame0145_labeled'
    # img_json_path = os.path.join(base_path, 'LeopardCamera1', '1635146368.574.json')
    # img_path = os.path.join(base_path, 'LeopardCamera1', '1635146368.574.png')

    # base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/tmp2'
    # base_path = '/home/zhangq/Desktop/zhangq/HM_RGB_Net/dataset/mydata_train'
    # base_path = '/media/ourDataset/v1.0_label/20211027_1_group0010_148frames_30labeled'
    base_path = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'

    folders = os.listdir(base_path)
    folders = sorted(folders)
    frame_num = -1
    for folder in folders:
        if 'labeled' not in folder:
            continue
        camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
        for file in os.listdir(camera_path):
            if file[-3:] == 'png':
                img_path = os.path.join(camera_path, file)
            if file[-4:] == 'json':
                img_json_path = os.path.join(camera_path, file)
        print('Img_Path:', img_path)
        # lidar_path = os.path.join(base_path, folder, 'VelodyneLidar')
        # for file in os.listdir(lidar_path):
        #     if file[-3:] == 'pcd':
        #         pcd_lidar = os.path.join(lidar_path, file)
        #     if file[-4:] == 'json':
        #         calib_lidar = os.path.join(lidar_path, file)
        # ti_path = os.path.join(base_path, folder, 'TIRadar')
        # for file in os.listdir(ti_path):
        #     if file[-3:] == 'pcd':
        #         TI_pcd_path = os.path.join(ti_path, file)
        #     if file[-4:] == 'json':
        #         TI_radar_json_path = os.path.join(ti_path, file)


        RGB_x, RGB_y, RGB_w, RGB_h = get_RGB_annotation(img_json_path)

        # dynamic_heatmap = cv2.imread(dynamic_heatmap_path)
        # dynamic_heatmap = dynamic_heatmap[143:, 180:361, :]
        # dynamic_heatmap = cv2.resize(dynamic_heatmap, (256, 256))
        # static_heatmap = cv2.imread(static_heatmap_path)
        # static_heatmap = static_heatmap[143:, 180:361, :]
        # static_heatmap = cv2.resize(static_heatmap, (256, 256))
        RGB_img = cv2.imread(img_path)


        # for v, u, tmp_l, tmp_w, yaw in zip(x, y, l, w, alpha):
        #     # Draw rotated box
        #     print(u, v, tmp_l, tmp_w, yaw)
        #     drawRotatedBox(dynamic_heatmap, u, v, tmp_w, tmp_l, np.pi-yaw, [0, 0, 0])
        #     drawRotatedBox(static_heatmap, u, v, tmp_w, tmp_l, np.pi-yaw, [0, 0, 0])

        # test = np.array([1810.4, 1023.3, 151.69999999999982, 94.0]).astype(int)
        # drawBox(RGB_img, test[0], test[1], test[2], test[3], [255, 0, 0])

        for u, v, w, h in zip(RGB_x, RGB_y, RGB_w, RGB_h):
            # print(u, v, w, h)
            drawBox(RGB_img, u, v, w, h, [0, 0, 255])

        cv2.imshow('RGB_img', cv2.resize(RGB_img, (1080, 720)))
        # cv2.imshow('RGB_img', RGB_img)
        if cv2.waitKey(0) & 0xFF == 27:
            break
        # cv2.imwrite('2DBox.jpg', RGB_img)
        # cv2.imshow('dynamic_heatmap', dynamic_heatmap)
        # # cv2.waitKey(0)
        # cv2.imshow('static_heatmap', static_heatmap)



        # hm = np.zeros((2, 429, 543), dtype=np.float32)
        # cls_idx = 1
        # down_sample_factor = 1
        # for v, u, tmp_l, tmp_w in zip(x, y, l, w):
        #     min_radius = 2
        #     if tmp_l > 0 and tmp_w > 0:
        #         radius = gaussian_radius((tmp_l, tmp_w), min_overlap=0.7)
        #         radius = max(min_radius, int(radius))
        #
        #         # coor_x, coor_y = center[0] / down_sample_factor, center[1] / down_sample_factor
        #         ct = np.array([u, v], dtype=np.float32)
        #         ct_int = ct.astype(np.int32)
        #         draw_umich_gaussian(hm[cls_idx], center=ct_int, radius=radius)
        #
        # plt.pcolor(hm[cls_idx])
        # plt.show()

        # cv2.waitKey(0)

if __name__ == "__main__":
    main()
