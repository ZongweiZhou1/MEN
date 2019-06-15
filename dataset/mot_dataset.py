import cv2
import sys
import json
import torch as th
import numpy as np
import torch.utils.data as data

from dataset.utils import random_pick
from dataset.box_helper import encoder
from dataset.utils import augment_horizontal_flip
from dataset.utils import augment_brightness_image

sys.path.append('../')


class MOT(data.Dataset):
    def __init__(self, file='MOT/dataset.json', train=True, max_num=6,
                 rangex=8, input_h=540, input_w=960, max_dis=16, stride=2):
        """
        :param file:        path of label file
        :param train:       trainset or valset
        :param max_num:     number of targets sampled from each frame
        :param rangex:      the maximum time margin of frame pairs
        :param input_H:     the input data's H
        :param input_W:     the input data's W
        """
        super(MOT, self).__init__()
        self.lmdb = json.load(open(file, 'r'))
        self.rangex = rangex
        self.train = train
        self.featstride = 4
        self.mean = np.expand_dims(np.expand_dims(np.array([109, 120, 119]),
                                   axis=0), axis=0).astype(np.float32)
        self.max_roi_num = max_num
        self.input_H, self.input_W = input_h, input_w
        self.grid_xy = self.get_grid_xy(max_dis, stride)

    def get_grid_xy(self, md, s):
        xx = np.arange(-md, md+1, s)
        yy = np.arange(md, md+1, s)
        grid_x, grid_y = np.meshgrid(xx, yy)
        grid_x = (grid_x / self.input_W) / self.featstride
        grid_y = (grid_y / self.input_H) / self.featstride
        grid_xy = np.concatenate((grid_x.reshape((-1, 1)), grid_y.reshape((-1, 1)),
                                  np.zeros((len(grid_x.flatten()), 2))), axis=1)
        return grid_xy

    def __getitem__(self, item):
        """get each sample
        :returns:
            imgs1:      array, [1, 3, H, W], the first image
            rois1:      array, [m, 4], normalized rois from im1
            clas1:      array, [n, 289], the corresponding classes label on each anchor of each roi
            regs1:      array, [n, 289, 4], the corresponding delta box on each anchor of each roi
            imgs2:      array, [1, 3, H, W], the second image
            rois2:      array, [m, 4], normalized rois from im1
            clas2:      array, [n, 289], the corresponding classes label on each anchor of each roi
            regs2:      array, [n, 289, 4], the corresponding delta box on each anchor of each roi
        """

        if self.train:
            id1 = self.lmdb['train_set'][item]
        else:
            id1 = self.lmdb['val_set'][item]
        range_up = self.lmdb['up_index'][id1]
        while True:
            # filter out the frame where there is no boxes
            x = np.arange(min(range_up, self.rangex))
            p = (x**2+1.0)/np.sum(x**2+1.0)
            id2 = random_pick(x, p) + id1
            if len(self.lmdb['pbbox'][id2]) > 0:
                break
        # read images
        imgs1 = cv2.imread(self.lmdb['img_paths'][id1]).astype(np.float32)
        imgs2 = cv2.imread(self.lmdb['img_paths'][id2]).astype(np.float32)
        if imgs1.shape[0] == 1080:
            imgs1 = cv2.resize(imgs1, dsize=(self.input_W, self.input_H)) - self.mean
            imgs2 = cv2.resize(imgs2, dsize=(self.input_W, self.input_H)) - self.mean
        else:
            desire_W = int((self.input_H/imgs1.shape[0]) * imgs1.shape[1])
            add_w = self.input_W - desire_W
            assert self.input_W > desire_W, 'errors in image"s size'
            # add_array = np.repeat(np.repeat(self.mean.reshape((1,1,3)), self.input_H, axis=0), add_w, axis=1)
            add_array = np.zeros((self.input_H, add_w, 3)).astype(np.float32)
            imgs1 = cv2.resize(imgs1, dsize=(desire_W, self.input_H)) - self.mean
            imgs2 = cv2.resize(imgs1, dsize=(desire_W, self.input_H)) - self.mean
            imgs1 = np.concatenate((imgs1, add_array), axis=1)
            imgs2 = np.concatenate((imgs2, add_array), axis=1)

        # select rois from different frames
        _rois1 = self.lmdb['pbbox'][id1]  # dict, track_id: (xmin, ymin, xmax, ymax) 'normalized'
        _rois2 = self.lmdb['pbbox'][id2]
        share_id = [k for k in _rois1 if k in _rois2]

        if len(share_id) == 0:
            imgs2    = imgs1.copy()
            _rois2   = _rois1
            share_id = [k for k in _rois1]

        share_id = np.random.choice(share_id, self.max_roi_num)
        rois1 = np.array([_rois1[idx] for idx in share_id])
        rois2 = np.array([_rois2[idx] for idx in share_id])
        # image and rois augments
        imgs1 = augment_brightness_image(imgs1)
        imgs2 = augment_brightness_image(imgs2)
        if np.random.uniform() > 0.5:
            imgs1, rois1 = augment_horizontal_flip(imgs1, rois1)
            imgs2, rois2 = augment_horizontal_flip(imgs2, rois2)

        clas1, regs1, clas2, regs2 = encoder(rois1, rois2, (self.input_H/self.featstride,
                                                            self.input_W/self.featstride),
                                             grid_xy=self.grid_xy)

        return imgs1.transpose((2, 0, 1)), rois1, imgs2.transpose((2, 0, 1)), rois2, clas1, regs1, clas2, regs2

    @staticmethod
    def collate_fn(batch):
        """collate_fn: list->batch
        :returns:
            imgs1:      FloatTensor, N x 3 x H x W
            rois1:      FloatTensor, N * max_num x 4
            imgs2:      FloatTensor, N x 3 x H x W
            rois2:      FloatTensor, N * max_num x 4
            clas1:      FloatTensor, N * max_num x 289
            regs1:      FloatTensor, N * max_num x 289 x 4
            clas2:      FloatTensor, N * max_num x 289
            regs2:      FloatTensor, N * max_num x 289 x 4
            indxs:      IntTensor, N * max_num, image index for roi align
        """
        ims1_list, rois1_list, ims2_list, rois2_list, cls1_list, regs1_list, \
        cls2_list, regs2_list = zip(*batch)
        imgs1 = th.FloatTensor(np.stack(ims1_list, axis=0)).contiguous()
        imgs2 = th.FloatTensor(np.stack(ims2_list, axis=0)).contiguous()
        rois1 = th.FloatTensor(np.concatenate(rois1_list, axis=0)).contiguous()
        rois2 = th.FloatTensor(np.concatenate(rois2_list, axis=0)).contiguous()
        clas1 = th.FloatTensor(np.concatenate(cls1_list, axis=0)).contiguous()
        regs1 = th.FloatTensor(np.concatenate(regs1_list, axis=0)).contiguous()
        clas2 = th.FloatTensor(np.concatenate(cls2_list, axis=0)).contiguous()
        regs2 = th.FloatTensor(np.concatenate(regs2_list, axis=0)).contiguous()
        indxs = np.array([[i] * len(rois1_list[0]) for i in range(len(ims1_list))]).flatten()
        indxs = th.IntTensor(indxs)

        return imgs1, rois1, imgs2, rois2, clas1, regs1, clas2, regs2, indxs

    def __len__(self):
        return len(self.lmdb['train_set']) if self.train else len(self.lmdb['val_set'])


# test unit
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torch.utils.data import DataLoader
    from dataset.box_helper import decoder

    data = MOT(train=True, rangex=8)
    n = len(data)
    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])

    dataloader = DataLoader(data, batch_size=1, shuffle=False, pin_memory=True, collate_fn=data.collate_fn,
                            num_workers=0, drop_last=True)
    mean = data.mean
    disp_list = []
    from tqdm import tqdm
    for i, v in enumerate(tqdm(dataloader)):
        imgs1 = v[0].numpy()[0]
        rois1 = v[1].numpy()  # N * max_num x 4
        imgs2 = v[2].numpy()[0]
        rois2 = v[3].numpy()

        clas1 = v[4].numpy()  # N * max_num x 289
        regs1 = v[5].numpy()
        clas2 = v[6].numpy()
        regs2 = v[7].numpy()
        indxs = v[8].numpy()  # N * max_num

        im1   = np.transpose(imgs1, (1, 2, 0)) + mean
        im2   = np.transpose(imgs2, (1, 2, 0)) + mean
        ims12 = np.concatenate((im1, im2), axis=1).astype(np.uint8)
        isize = im1.shape[:2]
        scale = np.array([isize[1], isize[0], isize[1], isize[0]]).astype(np.float32)
        ax.imshow(cv2.cvtColor(ims12, cv2.COLOR_BGR2RGB))

        prois2 = decoder(rois1, clas1, regs1, (isize[0]/4.0, isize[1]/4.0), grid_xy=data.grid_xy)
        prois1 = decoder(rois2, clas2, regs2, (isize[0]/4.0, isize[1]/4.0), grid_xy=data.grid_xy)

        for j in range(len(rois1)):
            w = rois1[j] * scale
            p = patches.Rectangle((w[0], w[1]), (w[2] - w[0]), (w[3] - w[1]), fill=False,
                                  clip_on=False, linewidth=2, edgecolor='r')
            ax.add_patch(p)
            w = prois1[j] * scale
            p = patches.Rectangle((w[0], w[1]), (w[2] - w[0]), (w[3] - w[1]), fill=False,
                                  clip_on=False, linewidth=1, edgecolor='g')
            ax.add_patch(p)

            w = rois2[j] * scale
            p = patches.Rectangle((w[0] + isize[1], w[1]), (w[2] - w[0]), (w[3] - w[1]), fill=False,
                                  clip_on=False, linewidth=2, edgecolor='r')
            ax.add_patch(p)

            w = prois2[j] * scale
            p = patches.Rectangle((w[0] + isize[1], w[1]), (w[2] - w[0]), (w[3] - w[1]), fill=False,
                                  clip_on=False, linewidth=1, edgecolor='g')
            ax.add_patch(p)
        plt.pause(3)
        plt.cla()

        # ims1 = v[0].numpy()
        # ims2 = v[1].numpy()
        # rois1 = v[2].numpy()
        # rois2 = v[3].numpy()
        # ids1 = v[4].numpy()
        # ids2 = v[5].numpy()
        # regs = v[6].numpy()
        # assert len(rois1) == len(rois2), 'rois should be appeared in pair'
        # assert len(rois1) == len(ids1), 'each roi should have an id'
        # assert len(rois2) == len(ids2), 'each roi should have an id'
        # assert len(rois1) == len(regs), 'each roi in rois corresponds a regression target'
        #
        # im1 = np.transpose((ims1[0] + mean), (1, 2, 0)).astype(np.uint8)
        # im2 = np.transpose((ims2[0] + mean), (1, 2, 0)).astype(np.uint8)
        # zx = np.concatenate((im1, im2), axis=1)
        # frame_sz = im1.shape[:2]
        # ax.imshow(cv2.cvtColor(zx, cv2.COLOR_BGR2RGB))
        # color = ['r', 'g']
        #
        # for j in range(len(rois1)):
        #     w = rois1[j]
        #     DW = frame_sz[1] * 0
        #     p = patches.Rectangle((w[0] + DW, w[1]), (w[2] - w[0]), (w[3] - w[1]),
        #                           fill=False, clip_on=False, linewidth=2, edgecolor=color[0])
        #     plt.text(w[0] + 5 + DW, w[1] + 5, '%d' % int(ids1[j]), fontsize=12)
        #     ax.add_patch(p)
        #     DW = frame_sz[1]
        #     p = patches.Rectangle((w[0] + DW, w[1]), (w[2] - w[0]), (w[3] - w[1]),
        #                           fill=False, clip_on=False, linewidth=2, edgecolor=color[0])
        #     plt.text(w[0] + 5 + DW, w[1] + 5, '%d' % int(ids1[j]), fontsize=12)
        #     ax.add_patch(p)
        #     w = rois2[j]
        #     p = patches.Rectangle((w[0] + DW, w[1]), (w[2] - w[0]), (w[3] - w[1]),
        #                           fill=False, clip_on=False, linewidth=2, edgecolor=color[1])
        #     plt.text(w[0] + 5 + DW, w[1] + 5, '%d' % int(ids1[j]), fontsize=12)
        #     ax.add_patch(p)
        #
        # plt.pause(3)
        # plt.cla()




