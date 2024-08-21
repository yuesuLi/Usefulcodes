import os

import numpy as np
from torch.utils.data import Dataset
import cv2
import json
import torch



def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        return json_data


class DatasetExpandGT(Dataset):
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root

        self.scenes = os.listdir(dataset_root)   # every frame path
        # self.scenes.remove('common_info')
        self.scenes.sort()
        self.lidar_data = []
        self.lidar_jsons = []

        for currScene in self.scenes:
            scene = os.path.join(dataset_root, currScene)

            lidar_path = os.path.join(os.path.join(scene, 'VelodyneLidar'))
            for file in os.listdir(lidar_path):
                if file[-4:] == 'json':
                    lidar_json_path = os.path.join(lidar_path, file)
                if file[-3:] == 'pcd':
                    Lidar_points_path = os.path.join(lidar_path, file)
            self.lidar_jsons.append(lidar_json_path)
            self.lidar_data.append(Lidar_points_path)

        assert len(self.lidar_data) == len(self.lidar_jsons)
        self.length = len(self.lidar_jsons)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # Lidar_points = read_pcd(self.lidar_data[item])  # x y z intensity idx_laser
        lidar_json = load_json(self.lidar_jsons[item])
        # curr_label = load_json(self.lidar_jsons[item])
        # lidar_annotation = load_json(self.lidar_jsons[item])['annotation']
        # with open(self.lidar_jsons[item]) as f:
        #     lidar_json = json.load(f)

        return lidar_json

if __name__ == '__main__':
    # data_dir = '/media/personal_data/zhangq/RadarRGBFusionNet2/dataset/datas/20221217_group0008_mode4_261frames'
    # label_dir ='/media/personal_data/zhangq/RadarRGBFusionNet2/dataset/labels/20221217_group0008_mode4_261frames'
    source = '/mnt/ourDataset_v2/ourDataset_v2/20221219_group0000_mode1_99frames'

    dataset = DatasetExpandGT(source)
    # lidar_annotation = dataset[0]
    # dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
    #                                                num_workers=1, drop_last=True)
    for frame_idx, lidar_json in enumerate(dataset):

        print('frame_Idx: ', frame_idx)
        print(lidar_json)

    # if cv2.waitKey(0) & 0xFF == 27:
    #     break

    print('done')