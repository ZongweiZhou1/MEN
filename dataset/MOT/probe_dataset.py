import os
import cv2
from os.path import join


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


if __name__ == '__main__':
    import numpy as np
    subsets = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
    data_base_path = '/data/zwzhou/Data'
    subset = 'MOT16-13'
    img_dir_path = join(data_base_path, 'MOT16/train/{}/img1/'.format(subset))
    img = cv2.imread(join(img_dir_path, '000001.jpg'))  # H x W x 3
    frame_sz = img.shape[:2]
    frames = read_mot_gt(join(data_base_path, 'MOT16/train/{}/gt/gt.txt'.format(subset)))
    n_frames = len(frames)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ws = []
    for f, frame in frames.items():
        img_path = join(img_dir_path, '{}.jpg'.format(str(f).zfill(6)))
        img = cv2.imread(img_path)
        for item in frame:
            tlwh = item[0]
            ws.append(item[0][2])
            if item[0][2] > 600:
                print(f)
            cc = (0, 0, 255)
            if item[0][2] > 600:
                cc = (0, 255, 0)
            cv2.rectangle(img, (int(tlwh[0]), int(tlwh[1])),(int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])),
                          cc, 2)
            cv2.putText(img, '%d'%item[0][2], (int(tlwh[0])+10, int(tlwh[1])+10), font, 1.2, (255, 0, 0), 2)
        cv2.imshow('frames_gt', img)
        cv2.waitKey(2)
    ws = np.array(ws)
    print(ws.max())
