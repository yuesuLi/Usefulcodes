import os
import json



def load_json(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
        return json_data



dataset_root = '/media/ourDataset/v1.0_label/20211026_2_group0003_77frames_16labeled'

scenes = os.listdir(dataset_root)   # every frame path
scenes.sort()
frame_num = -1
for scene in scenes:
    if 'labeled' not in scene:
        continue
    frame_num += 1
    camera_path = os.path.join(dataset_root, scene, 'LeopardCamera1')
    for file in os.listdir(camera_path):
        if file[-3:] == 'png':
            rgb_path = os.path.join(camera_path, file)
        if file[-4:] == 'json':
            rgb_json_path = os.path.join(camera_path, file)

        rgb_josn = load_json(rgb_json_path)
        rgb_annotations = rgb_josn['annotation']
        print('frame_num', frame_num)
        print('rgb_annotations', rgb_annotations, '\n')