import torch
import shutil
import numpy as np


def save_checkpoint(state, is_best, filename='checkpoint.pth', best_file='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)


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

def decoding(rois1, pred):
    # both Variable
    """ decode target
        :param:
            rois1:       Variable, [N, 4], (xmin, ymin, xmax, ymax)
            pred:        Variable, [N, 4], (dx, dy, dw, dh)
        :return:
            pred_tlbr:   Variable, [N, 4], (xmin, ymin, xmax, ymax)
        """
    cx_ = (rois1[..., 2] - rois1[..., 0]) * pred[..., 0] + \
          (rois1[..., 0] + rois1[..., 2]) / 2.0  # w*dx + cx
    cy_ = (rois1[..., 3] - rois1[..., 1]) * pred[..., 1] + \
          (rois1[..., 1] + rois1[..., 3]) / 2.0  # h*dy + cy
    w_ = torch.exp(pred[..., 2]) * (rois1[..., 2] - rois1[..., 0])  # exp(dw) * w
    h_ = torch.exp(pred[..., 3]) * (rois1[..., 3] - rois1[..., 1])  # exp(dh) * h

    pred_tlbr = torch.stack((cx_, cy_, w_, h_), dim=1)  # (cx, cy, w, h)
    pred_tlbr[:, :2] = pred_tlbr[:, :2] - pred_tlbr[:, 2:] / 2.0
    pred_tlbr[:, 2:] = pred_tlbr[:, 2:] + pred_tlbr[:, :2]
    return pred_tlbr  # (xmin, ymin, xmax, ymax)

