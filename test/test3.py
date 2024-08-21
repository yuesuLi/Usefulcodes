import os
import numpy as np
import math
import struct
import torch
from torch import nn
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import xlrd


# DataPathTrain = xlrd.open_workbook('/home/zhangq/Desktop/zhangq/LidarBEVTrack/DataPathTrain.xls')
# DataPathVal = xlrd.open_workbook('/home/zhangq/Desktop/zhangq/LidarBEVTrack/DataPathVal.xls')
# all_pathTrain = DataPathTrain.sheet_by_index(0).col_values(0)
# all_pathVal = DataPathVal.sheet_by_index(0).col_values(0)
# for GroupNum in range(len(all_pathTrain)):
#     base_path = os.path.join('/home/zhangq/Desktop/ourDataset/v1.0_label', all_pathTrain[GroupNum])
#     print('all_pathTrain:', base_path)
#
# print('********************************************************************************')
# for GroupNum in range(len(all_pathVal)):
#     base_path = os.path.join('/home/zhangq/Desktop/ourDataset/v1.0_label', all_pathVal[GroupNum])
#     print('all_pathTrain:', base_path)

# x_grid, y_grid = torch.meshgrid(torch.tensor(range(16)), torch.tensor(range(32)))
# xy_grid = [x_grid, y_grid]
# xy_grid = torch.unsqueeze(torch.stack(xy_grid, axis=-1), 2)
# xy_grid = torch.tile(torch.unsqueeze(xy_grid, 0), \
#                 [1, 1, 1, 6, 1])
# xy_grid = xy_grid.float()

# a = torch.tensor([[[0.7,0.2,0.1], [0.7,0.2,0.1]],[[0.7,0.2,0.1], [0.7,0.2,0.1]]])
# b = torch.tensor([[[0.99,0.01,0.01], [0.99,0.01,0.01]], [[0.99,0.01,0.01], [0.99,0.01,0.01]]])
# b = torch.tensor([[[0.99,0.01,0.01], [0.01,0.99,0.99]], [[0.01,0.991,0.91], [0.99,0.01,0.01]]])
# # a = torch.tensor([[[0.7,0.2,0.1],[0.7,0.2,0.1]]])
# # b = torch.tensor([[[1.,0,0], [1.,0,0]]])
# loss = nn.CrossEntropyLoss()(a,b)
# # loss = nn.CrossEntropyLoss(reduction='sum')(a, b)
# print('loss', loss)

a = torch.tensor(np.arange(100).reshape((2, 5, 10)))
b = torch.tensor(np.arange(100).reshape((2, 5, 10)))
b = -2 * b
c = torch.tensor(np.arange(10).reshape((2, 5, 1)))
d = a-b
e = c*d

print(' ')