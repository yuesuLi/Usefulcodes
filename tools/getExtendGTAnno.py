
import os
import sys
import argparse
import openpyxl
import pandas as pd
import json

CURRENT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
ROOT = os.path.join(CURRENT_ROOT, '../')
sys.path.append(ROOT)


def log(text):
    print(text)

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def getAnno(currFrame, outputGroupPath, frameName):

    outputFramePath = os.path.join(outputGroupPath, frameName)
    if not os.path.exists(outputFramePath):
        os.makedirs(outputFramePath)
        # log('create {}'.format(outputFramePath))
    extendGTJsonPath = os.path.join(outputFramePath, 'extendGT.json')
    extendAnno = currFrame.to_dict('records')
    for anno in extendAnno:
        anno['object_id'] = str(anno['object_id'])
    with open(extendGTJsonPath, "w") as f:
        json.dump(extendAnno, f, indent=4)
        # json_data = json.dumps(withAnno)
        # f.write(json_data)
        # print("加载入文件完成...")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--groups_xlsx_path', type=str, default='/mnt/ChillDisk/personal_data/zhangq/RadarRGBFusionNet2/GroupPath/20231113AllData.xlsx', help='groups name')
    parser.add_argument('--extendGT_xlsx_base_path', type=str, default='/mnt/ChillDisk/personal_data/zhangq/UsefulCode/tools/runs3', help='dataset base path')
    parser.add_argument('--output_foldername', type=str, default='runs', help='output folder name')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    xlsx_path = args.groups_xlsx_path
    extendGT_base_path = args.extendGT_xlsx_base_path
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
        extendGT_path = os.path.join(extendGT_base_path, groupname, 'groupExtendGTs.xlsx')
        if os.path.exists(extendGT_path):
            output_group_path = os.path.join(root_output, groupname)
            if not os.path.exists(output_group_path):
                os.makedirs(output_group_path)
                log('create {}'.format(output_group_path))

                extendGT = pd.read_excel(extendGT_path, index_col=0, engine='openpyxl')
                all_frame = list(set(extendGT['framename']))
                all_frame.sort()
                for j, framename in enumerate(all_frame):
                    # log('   {}/{} {}'.format(j + 1, len(all_frame), framename))
                    curr_frame = extendGT[extendGT['framename'] == framename]
                    getAnno(curr_frame, output_group_path, framename)
            else:
                continue


        else:
            raise FileNotFoundError("file not found")



    print()


if __name__ == '__main__':
    main()













