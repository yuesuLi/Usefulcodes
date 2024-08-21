import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import json
import os
import cv2



# base_path = '/media/ourDataset/v1.0_label/20211027_1_group0026_144frames_29labeled'
base_path = '/media/personal_data/zhangq/TransT-main/dataset/mydata'
# RGB_path = '/media/personal_data/zhangq/TransT-main/dataset/mydata/20211027_1_group0021_frame0001/TIRadar/dynamic_heatmap_1635319097.510.png'
# heatmap_path = '/media/personal_data/zhangq/TransT-main/dataset/mydata/20211027_1_group0021_frame0001/TIRadar/dynamic_heatmap_1635319097.510.png'
# save_path = './save/test/'

# cv2.namedWindow('imgs', 0)
# cv2.resizeWindow('imgs', 1600, 900)

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
    ti_path = os.path.join(base_path, folder, 'TIRadar')
    for file in os.listdir(ti_path):
        if file[0] == 'd':
            dynamic_heatmap_path = os.path.join(ti_path, file)
        if file[0] == 's':
            static_heatmap_path = os.path.join(ti_path, file)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (512, 512))
    cv2.imshow('RGB', img)
    cv2.waitKey(0)

    dynamic_heatmap = cv2.imread(dynamic_heatmap_path)
    dynamic_heatmap = cv2.resize(dynamic_heatmap, (256, 256))
    cv2.imshow('dynamic_heatmap', dynamic_heatmap)
    cv2.waitKey(0)

    static_heatmap = cv2.imread(static_heatmap_path)
    static_heatmap = cv2.resize(static_heatmap, (256, 256))
    cv2.imshow('static_heatmap', static_heatmap)
    cv2.waitKey(0)








# anno_path = '/media/personal_data/zhangq/TransT-main/dataset/OTB2015/OTB.json'
# with open(anno_path, 'r') as f:
#     dataset = json.load(f)  # dict, dataset['Basketball']:dict
#
# print('dataset:', type(dataset['Basketball']))
# img_names = dataset['Basketball']['img_names']  # list
# gt_rect = dataset['Basketball']['gt_rect']  # list
# frame_id = int('0664')
# print('img_names:', img_names[frame_id])
# print('gt_rect:', gt_rect[frame_id])
#
# print('test:', int('002'))



# X_bins = X_bins.reshape((X_bins.shape[0] * X_bins.shape[1],))
# Y_bins = Y_bins.reshape((Y_bins.shape[0] * Y_bins.shape[1],))
# TI_heatmap_dynamic = TI_heatmap_dynamic.reshape((TI_heatmap_dynamic.shape[0] * TI_heatmap_dynamic.shape[1],))
# df = get_data(mmxx=X_bins, mmyy=Y_bins, mmzz=TI_heatmap_dynamic)
# layout = go.Layout(
#     # plot_bgcolor='red',  # 图背景颜色
#     paper_bgcolor='white',  # 图像背景颜色
#     autosize=True,
#     title='T2热力图',
#     titlefont=dict(size=30, color='gray'),
#
#     # 图例相对于左下角的位置
#     legend=dict(
#         x=0.02,
#         y=0.02
#     )
# )
#
# fig = go.Figure(data=go.Heatmap(
#     showlegend=True,
#     name='Value',
#     x=df['x'],
#     y=df['y'],
#     z=df['z'],
#     type='heatmap',
# ),
#     layout=layout
# )
#
# fig.update_layout(margin=dict(t=100, r=150, b=100, l=100), autosize=True)
# pio.write_image(fig, '/media/personal_data/zhangq/UsefulCode/1.jpg')
# # fig.show()
