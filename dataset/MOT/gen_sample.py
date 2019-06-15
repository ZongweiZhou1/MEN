from os.path import join
import numpy as np
import time
import json
from tqdm import tqdm as tqdm
import os
import cv2


np.random.seed(512)

"""
labels = {
'pedestrian', ...        %1
'person_on_vhcl', ...    %2
'car', ...               %3
'bicycle', ...           %4
'mbike', ...             %5
'non_mot_vhcl', ...      %6
'static_person', ...     %7
'distractor', ...        %8
'occluder', ...          %9
'occluder_on_grnd', ...  %10
'occluder_full',         %11
'reflection', ...        %12
'crowd' ...              %13
}; 
"""


def read_mot_gt(filename):
    labels = {1, 2, 7}
    targets = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                targets.setdefault(fid, list())
                label = int(float(linelist[-2])) if len(linelist) > 7 else -1
                if label not in labels:
                    continue
                if float(linelist[-3]) == 0:  # ignored
                    continue
                if float(linelist[-1]) < 0.4:  # visibility
                    continue

                tlwh = tuple(map(float, linelist[2:7]))
                target_id = int(linelist[1])

                targets[fid].append((tlwh, target_id, float(linelist[-1])))
    sort_targets = dict()  # important
    for k in sorted(targets.keys()):
        sort_targets[k] = targets[k]
    return sort_targets


subsets = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
data_base_path = '/data/zwzhou/Data'

padding = 0.2  # padding on w and h
num_all_frames = 5316
num_val = 200
lmdb = dict()
lmdb['up_index'] = np.zeros(num_all_frames, np.int)  # there are how many frames left in current video

count = 0
begin_time = time.time()
img_paths = []
gtboxes = []
boxes_num = np.zeros(num_all_frames, np.int)
unique_trackid = {}
track_id = 0
for sid, subset in enumerate(subsets):
    img_dir_path = join(data_base_path, 'MOT16/train/{}/img1/'.format(subset))
    img = cv2.imread(join(img_dir_path, '000001.jpg'))   # H x W x 3
    frame_sz = img.shape[:2]
    frames = read_mot_gt(join(data_base_path, 'MOT16/train/{}/gt/gt.txt'.format(subset)))
    n_frames = len(frames)
    print(subset+'  >>>')
    for f, frame in tqdm(frames.items()):
        img_paths.append(join(img_dir_path, '{}.jpg'.format(str(f).zfill(6))))
        bbox = dict()
        for item in frame:
            tlwh = item[0]
            if (sid, item[1]) not in unique_trackid:
                unique_trackid[(sid, item[1])] = track_id
                track_id += 1  # the number of identities
            # (xmin, ymin, xmax, ymax) normalized
            bbox[unique_trackid[(sid, item[1])]] = [tlwh[0]/frame_sz[1],
                                                    tlwh[1]/frame_sz[0],
                                                    (tlwh[0] + tlwh[2])/frame_sz[1],
                                                    (tlwh[1] + tlwh[3])/frame_sz[0]]
            # the last value is the idx of sequence


        gtboxes.append(bbox)
        boxes_num[count] = len(bbox)

        lmdb['up_index'][count] = n_frames - f
        count += 1

print('There are %d classes'%track_id)
template_id = np.where(np.logical_and(lmdb['up_index'] > 1, boxes_num > 0))[0]
rand_split = np.random.permutation(len(template_id))
lmdb['train_set'] = template_id[rand_split[:(len(template_id)-num_val)]].tolist()
lmdb['val_set'] = template_id[rand_split[(len(template_id)-num_val):]].tolist()
lmdb['up_index'] = lmdb['up_index'].tolist()
lmdb['img_paths'] = img_paths
lmdb['pbbox'] = gtboxes

print('save lmdb json file for MOT training sets')
json.dump(lmdb, open('dataset.json', 'w'), indent=2)
print('done!')