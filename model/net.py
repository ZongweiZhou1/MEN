import torch.nn as nn

from model.hourglass import BackBone
from model.modules import trackerNet, reidNet


class MENNet(nn.Module):  # Reid and Track network
    def __init__(self, nclasses=462, max_displacement=16, stride=2, featstride=4):
        super(MENNet, self).__init__()
        self.nclasses = nclasses
        self.backbone = BackBone()
        self.reid = reidNet(nclasses=nclasses, featstride=featstride)
        self.track = trackerNet(max_disp=max_displacement, stride=stride)


    def forward(self, ims1, ims2, rois1, rois_index1, rois2=None, rois2_index2=None):
        ims1_feat = self.backbone(ims1)
        ims2_feat = self.backbone(ims2)

        track_res = self.track(ims1_feat, ims2_feat, rois1, rois_index1, rois2, rois_index2)
        # N x 289 x 2, N x 289 x 4
        if self.nclasses > 0:
            ims1_feats, ims1_logits = self.reid(ims1_feat, rois1, rois_index1)
            ims2_feats, ims2_logits = self.reid(ims2_feat, rois2, rois_index2)
            reid_res = [ims1_feats, ims1_logits, ims2_feats, ims2_logits]
        else:
            ims1_feats = self.reid(ims1_feat, rois1, rois_index1)
            ims2_feats = self.reid(ims2_feat, rois2, rois_index2)
            reid_res = [ims1_feats, ims2_feats]

        return track_res, reid_res


if __name__ == '__main__':
    import torch
    s1, s2 = 360, 480
    ims1 = torch.rand(1, 3, s1, s2).cuda()
    ims2 = torch.rand(1, 3, s1, s2).cuda()
    rois1 = torch.FloatTensor([[30/s1, 30/s2, 80/s1, 60/s2], [35/s1, 35/s2, 76/s1, 62/s2]]).cuda()
    rois2 = torch.FloatTensor([[45/s1, 50/s2, 120/s1, 93/s2], [30/s1, 30/s2, 75/s1, 60/s2]]).cuda()
    rois_index1 = torch.IntTensor([0, 0]).cuda()
    rois_index2 = torch.IntTensor([0, 0]).cuda()
    net = MENNet(nclasses=80).cuda()
    track_res, reid_res = net(ims1, ims2, rois1, rois_index1, rois2, rois_index2)
    cls_logits, reg_boxes = track_res[0], track_res[1]

    print(cls_logits.size())
    print(reg_boxes.size())
    ims1_feats, ims1_logits_list = reid_res[0], reid_res[1]
    print(len(ims1_feats))
    print(ims1_feats.size())
    print(len(ims1_logits_list))
    print(ims1_logits_list[0].size())

