# /mnt/ourDataset_v2/ourDataset_v2/20221219_group0000_mode1_99frames/frame0001/VelodyneLidar

import glob
import os
import sys
import argparse
import openpyxl

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)

import numpy as np
import pandas as pd
import json

def log(text):
    print(text)

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

# with open("record.json", "w") as f:
#     # json.dump(withAnno, f, indent=4)
#     # json_data = json.dumps(withAnno)
#     # f.write(json_data)
#     print("加载入文件完成...")

def generate_interval_gt(df_group, preFrame, currFrame, groupname):
    preNum = int(preFrame['framename'][5:])
    # frame_len = len(str(preNum))
    currNum = int(currFrame['framename'][5:])

    addLabels = []
    for intervalNum in range(preNum+1, currNum):
        addLabel = {}
        frameNumStr = preFrame['framename'][0:-len(str(intervalNum))] + str(intervalNum)
        addLabel['alpha'] = preFrame['alpha'] + (currFrame['alpha'] - preFrame['alpha']) / 5 * (intervalNum % 5)
        addLabel['class'] = preFrame['class']
        addLabel['h'] = (preFrame['h'] + currFrame['h']) / 2
        addLabel['l'] = (preFrame['l'] + currFrame['l']) / 2
        addLabel['motion'] = preFrame['motion']
        addLabel['object_id'] = preFrame['object_id']
        addLabel['occluded'] = preFrame['occluded']
        addLabel['truncated'] = preFrame['truncated']
        addLabel['w'] = (preFrame['w'] + currFrame['w']) / 2
        addLabel['x'] = preFrame['x'] + (currFrame['x'] - preFrame['x']) / 5 * (intervalNum % 5)
        addLabel['y'] = preFrame['y'] + (currFrame['y'] - preFrame['y']) / 5 * (intervalNum % 5)
        addLabel['z'] = preFrame['z'] + (currFrame['z'] - preFrame['z']) / 5 * (intervalNum % 5)
        addLabel['framename'] = frameNumStr
        addLabel['groupname'] = groupname
        addLabels.append(addLabel)

    df_frame = pd.DataFrame(addLabels)
    df_group = pd.concat((df_group, df_frame))

    return df_group

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--groups_xlsx_path', type=str, default='/mnt/ChillDisk/personal_data/zhangq/RadarRGBFusionNet2/GroupPath/20231113AllData.xlsx', help='groups name')
    parser.add_argument('--dataset_base_path', type=str, default='/mnt/ourDataset_v2/ourDataset_v2_label', help='dataset base path')
    parser.add_argument('--output_foldername', type=str, default='runs3', help='output folder name')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    xlsx_path = args.groups_xlsx_path
    dataset_base_path = args.dataset_base_path
    output_foldername = args.output_foldername

    DataPath = openpyxl.load_workbook(xlsx_path)
    ws = DataPath.active
    groups_excel = ws['A']
    groupnames = []
    for cell in groups_excel:
        if not cell.value:
            continue
        groupnames.append(cell.value)
    groups_length = len(groupnames)

    root_output = os.path.join(CURRENT_ROOT, output_foldername)
    if not os.path.exists(root_output):
        os.makedirs(root_output)
        log('create {}'.format(root_output))

    for i, groupname in enumerate(groupnames):

        # log('=' * 100)
        log('{}/{} {}'.format(i + 1, groups_length, groupname))
        output_folder = os.path.join(root_output, groupname)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
            log('create {}'.format(output_folder))

        # data generate
        df_group_path = os.path.join(output_folder, 'groupGTs.xlsx')
        if os.path.exists(df_group_path):
            continue
            df_group = pd.read_excel(df_group_path, index_col=0, engine='openpyxl')
        else:
            df_group = pd.DataFrame()
            group_folder = os.path.join(dataset_base_path, groupname)
            framenames = os.listdir(group_folder)
            framenames = [framename.split('.')[0] for framename in framenames]
            framenames.sort()
            for j, framename in enumerate(framenames):
                # log('   {}/{} {}'.format(j + 1, len(framenames), framename))

                scene = os.path.join(group_folder, framename)
                lidar_path = os.path.join(os.path.join(scene, 'VelodyneLidar'))
                for file in os.listdir(lidar_path):
                    if file[-4:] == 'json':
                        label_path = os.path.join(lidar_path, file)
                label = load_json(label_path)['annotation']
                for anno in label:
                    anno['object_id'] = int(anno['object_id'])
                df_frame = pd.DataFrame(label)

                df_frame['framename'] = framename
                df_group = pd.concat((df_group, df_frame))
            df_group['groupname'] = groupname
            df_group = df_group.sort_values(by=['framename', 'object_id'], ascending=[True, True])
            df_group.to_excel(df_group_path)

        # data process
        df_group = df_group.sort_values(by=['object_id', 'framename'], ascending=[True, True])
        all_id = list(set(df_group['object_id']))
        all_id.sort()
        for id in all_id:
            own_id_frames = df_group[df_group['object_id'] == id]
            own_id_frames_length = len(own_id_frames)
            for frame_num in range(1, own_id_frames_length):
                pre_frame = own_id_frames.iloc[frame_num-1]
                curr_frame = own_id_frames.iloc[frame_num]
                frame_interval = int(curr_frame['framename'][5:]) - int(pre_frame['framename'][5:])
                if frame_interval == 5:
                    df_group = generate_interval_gt(df_group, pre_frame, curr_frame, groupname)

        df_extend_group_path = os.path.join(output_folder, 'groupExtendGTs.xlsx')
        df_group = df_group.sort_values(by=['framename', 'object_id'], ascending=[True, True])
        df_group.to_excel(df_extend_group_path)
        print()






if __name__ == '__main__':
    main()