import numpy as np
from dataset.utils import augment_jitter_rois

def tlbr2xywh(tlbr):
    xywh = tlbr.copy()
    xywh[..., 2:] -= xywh[..., :2]
    xywh[..., :2] += xywh[..., 2:]/2.0
    return xywh

def xywh2tlbr(xywh):
    tlbr = xywh.copy()
    tlbr[..., :2] -= tlbr[..., 2:]/2.0
    tlbr[..., 2:] += tlbr[..., :2]
    return tlbr

def IoUs(tlbrs1, tlbrs2):
    xmin = np.maximum(tlbrs1[..., 0], tlbrs2[..., 0])
    ymin = np.maximum(tlbrs1[..., 1], tlbrs2[..., 1])
    xmax = np.minimum(tlbrs1[..., 2], tlbrs2[..., 2])
    ymax = np.minimum(tlbrs1[..., 3], tlbrs2[..., 3])

    dx = np.maximum(0, xmax-xmin+1)  # N x 289
    dy = np.maximum(0, ymax-ymin+1)

    unions = (tlbrs1[..., 2] - tlbrs1[..., 0] + 1) * (tlbrs1[..., 3]- tlbrs1[..., 1] + 1) + \
            (tlbrs2[..., 2] - tlbrs2[..., 0] + 1) * (tlbrs2[..., 3] - tlbrs2[..., 1] + 1)
    ious = dx * dy / (unions - dx * dy + 1e-12)
    return ious  # N x 289

def encoder(rois1, rois2, input_size, grid_xy=None, train=True):
    """Encoder
    :param rois1:           N x 4, array, normalized, [x1, y1, x2, y2]
    :param rois2:           N x 4, array, normalized
    :param input_size:      2, array or list, [h,w] of the network output
    :param grid_xy:         the grid xy of candidated regions
    :return: cls1, deltas1, cls2, deltas2
    """
    assert len(rois1) == len(rois2), 'shape of input #1 and input #2 should be consistent.'
    if grid_xy is None:
        xx = np.arange(-16, 17, 2)
        yy = np.arange(-16, 17, 2)
        grid_x, grid_y = np.meshgrid(xx, yy)
        grid_x = grid_x / input_size[1]
        grid_y = grid_y / input_size[0]
        grid_xy = np.concatenate((grid_x.reshape((-1, 1)), grid_y.reshape((-1, 1)),
                                  np.zeros((len(grid_x.flatten()), 2))), axis=1)

    # xywh1, xywh2 = tlbr2xywh(rois1), tlbr2xywh(rois2)
    if train:
        xywh1, xywh2 = tlbr2xywh(augment_jitter_rois(rois1)), tlbr2xywh(augment_jitter_rois(rois2))
    else:
        xywh1, xywh2 = tlbr2xywh(rois1), tlbr2xywh(rois2)
    grid_xy = grid_xy[np.newaxis, :, :]
    xywh1[:, 2] = np.ceil(xywh1[:, 2] * input_size[1] / 10) * 10 / input_size[1]
    xywh1[:, 3] = np.ceil(xywh1[:, 3] * input_size[0] / 10) * 10 / input_size[0]
    xywh2[:, 2] = np.ceil(xywh2[:, 2] * input_size[1] / 10) * 10 / input_size[1]
    xywh2[:, 3] = np.ceil(xywh2[:, 3] * input_size[0] / 10) * 10 / input_size[0]

    anchors_xywh1 = xywh1[:, np.newaxis, :] + grid_xy
    anchors_xywh2 = xywh2[:, np.newaxis, :] + grid_xy
    anchors_tlbr1 = xywh2tlbr(anchors_xywh1)
    anchors_tlbr2 = xywh2tlbr(anchors_xywh2)

    xywh1_, xywh2_ = tlbr2xywh(rois1), tlbr2xywh(rois2)
    # delta, cls
    # rois1
    rois1_ious = IoUs(rois2[:, np.newaxis, :], anchors_tlbr1)  # N x 289
    cls1 = -np.ones(rois1_ious.shape)
    cls1[rois1_ious > 0.7] = 1
    cls1[rois1_ious < 0.4] = 0
    max_ious_idx = np.argmax(rois1_ious, axis=1)
    cls1[np.arange(len(max_ious_idx)), max_ious_idx] = 1  # N x 289

    delta_x = (xywh2_[:, np.newaxis, 0] - anchors_xywh1[..., 0])/anchors_xywh1[..., 2]
    delta_y = (xywh2_[:, np.newaxis, 1] - anchors_xywh1[..., 1])/anchors_xywh1[..., 3]
    delta_w = np.log(xywh2_[:, np.newaxis, 2] / anchors_xywh1[..., 2])
    delta_h = np.log(xywh2_[:, np.newaxis, 3] / anchors_xywh1[..., 3])
    deltas1 = np.stack((delta_x, delta_y, delta_w, delta_h), axis=2)  # N x 289 x 4
    # rois2
    rois2_ious = IoUs(rois1[:, np.newaxis, :], anchors_tlbr2)  # N x 289
    cls2 = -np.ones(rois2_ious.shape)
    cls2[rois2_ious > 0.7] = 1
    cls2[rois2_ious < 0.4] = 0
    max_ious_idx = np.argmax(rois2_ious, axis=1)
    cls1[np.arange(len(max_ious_idx)), max_ious_idx] = 1  # N x 289

    delta_x = (xywh1_[:, np.newaxis, 0] - anchors_xywh2[..., 0]) / anchors_xywh2[..., 2]
    delta_y = (xywh1_[:, np.newaxis, 1] - anchors_xywh2[..., 1]) / anchors_xywh2[..., 3]
    delta_w = np.log(xywh1_[:, np.newaxis, 2] / anchors_xywh2[..., 2])
    delta_h = np.log(xywh1_[:, np.newaxis, 3] / anchors_xywh2[..., 3])
    deltas2 = np.stack((delta_x, delta_y, delta_w, delta_h), axis=2)  # N x 289 x 4

    return cls1, deltas1, cls2, deltas2


def decoder(rois, cls, deltas, input_size, grid_xy=None):
    """
    :param rois:        N x 4
    :param cls:         N x 289
    :param deltas:      N x 289 x 4
    :param input_size:  [h, w]
    :param grid_xy:
    :return:  prois (predict rois), in `tlbr` style
    """
    if grid_xy is None:
        xx = np.arange(-16, 17, 2)
        yy = np.arange(-16, 17, 2)
        grid_x, grid_y = np.meshgrid(xx, yy)
        grid_x = grid_x / input_size[1]
        grid_y = grid_y / input_size[0]
        grid_xy = np.concatenate((grid_x.reshape((-1, 1)), grid_y.reshape((-1, 1)),
                                  np.zeros((len(grid_x.flatten()), 2))), axis=1)
    xywh = tlbr2xywh(rois)
    grid_xy = grid_xy[np.newaxis, :, :]
    xywh[:, 2] = np.ceil(xywh[:, 2] * input_size[1] / 10) * 10 / input_size[1]
    xywh[:, 3] = np.ceil(xywh[:, 3] * input_size[0] / 10) * 10 / input_size[0]
    anchors_xywh = xywh[:, np.newaxis, :] + grid_xy  # N x 289 x 4
    select_anchor = np.argmax(cls, axis=1)
    select_anchor_xywh = anchors_xywh[np.arange(len(select_anchor)), select_anchor]  # N x 4
    #
    delta_per_sample = deltas[np.arange(len(select_anchor)), select_anchor]  # N x 4
    dx = delta_per_sample[:, 0] * select_anchor_xywh[:, 2] + select_anchor_xywh[:, 0]
    dy = delta_per_sample[:, 1] * select_anchor_xywh[:, 3] + select_anchor_xywh[:, 1]
    dw = np.exp(delta_per_sample[:, 2]) * select_anchor_xywh[:, 2]
    dh = np.exp(delta_per_sample[:, 3]) * select_anchor_xywh[:, 3]

    return xywh2tlbr(np.stack((dx, dy, dw, dh), axis=1))


# test unit
if __name__ == '__main__':
    # tlbr2xywh
    tlbr = np.array([[30, 50, 80, 90],[30, 20, 50,70]]).astype(np.float32)
    xywh = tlbr2xywh(tlbr)
    print('xywh:', xywh)
    # xywh2tlbr
    tlbr = xywh2tlbr(xywh)
    print('tlbr:', tlbr)
    # IoUs
    tlbr1 = np.array([[30, 50, 80, 90], [30, 20, 50, 70]])
    tlbr2 = np.array([[35, 55, 45, 75], [30, 20, 50, 70], [80,130,120, 180]])
    ious = IoUs(tlbr1[:, np.newaxis, :], tlbr2[np.newaxis, :, :])
    print('ious`shape:', ious.shape)
    print('ious:')
    print(ious)
    # encoder
    input_size = np.array([120, 100]).astype(np.float32)  # h, w
    tlbr1 = tlbr1/input_size[0]   # normalize
    tlbr2 = tlbr2[:2]/input_size[1]

    cls1, deltas1, cls2, deltas2 = encoder(tlbr1, tlbr2, input_size)
    print('cls1 shape:', cls1.shape)
    print('cls2 shape:', cls2.shape)
    print('delta1 shape:', deltas1.shape)
    print('delta2 shape:', deltas2.shape)
    # decoder
    ptlbr2 = decoder(tlbr1, cls1, deltas1, input_size)
    print('predict tlbr2 shape:', ptlbr2.shape)
    print('predict tlbr2:')
    print(ptlbr2)
    print('ground truth tlbr2:')
    print(tlbr2)








