import torch
import torch.nn as nn
import numpy as np
import random
import sys
from bbox_helper import xywh2tlbr
import torch.nn.functional as F

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

def load_pretrain(net, param_dict, strict=True):
    prefix = ''
    if param_dict.keys()[0].startswith('module.') and \
           not net.state_dict().keys()[0].startswith('module.'):
        prefix = 'module.'

    if strict:
        net_keys = set([prefix+k for k in net.state_dict().keys()])
        param_keys = set(param_dict.keys())
        keys_only_in_net = net_keys - param_keys
        keys_only_in_dict = param_keys - net_keys
        if len(keys_only_in_dict) > 0 or len(keys_only_in_net) > 0:
            print('keys only in net:')
            print(keys_only_in_net)
            print('\n keys only in dict')
            print(keys_only_in_dict)
            raise("The params are not consistency!")

    for k, v in net.state_dict().items():
        k = prefix + k
        if k in param_dict:
            param = torch.from_numpy(np.asarray(param_dict[k]))
            if v.size() != param.size():
                print('{} Inconsistent shape: {}, {}'.format(k, v.size(), param.size()))
            else:
                v.copy_(param)
        else:
            print('No Layer: {} in the param dict'.format(k))

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_checkpoint(state, filename):
    torch.save(state, filename)


def _smooth_l1_loss(bbox_pred, bbox_targets, sigma=1):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    abs_box_diff = torch.abs(box_diff)
    smoothL1_sign = (abs_box_diff < 1. / sigma_2).detach().float()
    loss_box = torch.pow(box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss_box = loss_box.sum()
    return loss_box

def multi_reg_loss(bbox_pred, bbox_targets):
    reg_loss = _smooth_l1_loss(bbox_pred[:, :-1], bbox_targets[:, :-1])
    vis_loss = F.smooth_l1_loss(bbox_pred[:, -1:], bbox_targets[:, -1:])
    return reg_loss, vis_loss

def weight_l1_loss(bbox_pred, bbox_targets):
    diff = (bbox_pred - bbox_targets).abs()
    diff = diff.mean(dim=1, keepdim=False)  # N
    loss = diff.mean()
    return loss

def match_metric(rois, preds):
    """all the input are np.array, return the metric evaluationg the matching performance,
     here the mean Iou is adapted, rois and preds are all [xmin, ymin, xmax, ymax]"""
    xmin = np.maximum(rois[:, 0], preds[:, 0])
    xmax = np.minimum(rois[:, 2], preds[:, 2])
    ymin = np.maximum(rois[:, 1], preds[:, 1])
    ymax = np.maximum(rois[:, 3], preds[:, 3])
    inter = np.maximum(xmax - xmin + 1, 0) * np.maximum(ymax - ymin + 1, 0) * 1.

    area_rois = (rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1)
    area_pred = (preds[:, 2] - preds[:, 0] + 1) * (preds[:, 3] - preds[:, 1] + 1)

    iou = inter / (area_pred + area_rois - inter)
    acc = np.mean(iou)
    return acc


if __name__ == '__main__':
    from torch.autograd import Variable
    bbox_pred = Variable(torch.rand(5, 4))
    bbox_targets = Variable(torch.rand(5, 4))
    flags = Variable(torch.FloatTensor([0,1,1,1,0]))
    print(_smooth_l1_loss(bbox_pred, bbox_targets, flags))
    print(weight_l1_loss(bbox_pred, bbox_targets, flags))

    print(match_metric(bbox_pred, bbox_targets))

