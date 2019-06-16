import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model.hourglass import residual, convolution
from model.roialign_package.roi_align.crop_and_resize import CropAndResize as RoIAlign
from model.correlation_package.correlation import Correlation


class trackerNet(nn.Module):
    def __init__(self, input_dim=256, max_disp=16, stride=2):
        super(trackerNet, self).__init__()
        self.cnv = nn.Sequential(convolution(3, input_dim, input_dim*2),
                                 residual(3, input_dim*2, input_dim*2),
                                 convolution(3, input_dim*2, input_dim),
                                 nn.Conv2d(input_dim, input_dim, 1))

        self.conv_corr = Correlation(pad_size=max_disp, kernel_size=1, max_displacement=max_disp,
                                     stride1=1, stride2=stride)
        self.roipooling = RoIAlign(16, 16)

        in_feat = (2 * int(max_disp / stride) + 1) ** 2
        # class
        self.cls_conv = convolution(3, in_feat, in_feat)
        self.cls      = nn.Linear(16**2, 1)

        # regression
        self.reg_conv = convolution(3, in_feat, in_feat)
        self.reg      = nn.Linear(16**2, 4)

    def forward(self, ims1, ims2, rois1, rois_index1, rois2=None, rois_index2=None):
        """
        :param ims1:   N x 256 x H x W
        :param ims2:   N x 256 x H x W
        :param rois1:  N x 4
        :param rois_index1:  N, IntTensor
        :param rois2:  N x 4
        :param rois_index2:  N, IntTensor
        :return:
        """
        feat1 = self.cnv(ims1)
        feat2 = self.cnv(ims2)
        corr_feat = self.conv_corr(feat1, feat2)

        roi_feats = self.roipooling(corr_feat, rois1, rois_index1)
        cls_feats = self.cls(self.cls_conv(roi_feats).view(roi_feats.size(0)*roi_feats.size(1), -1).contiguous())
        cls_feats = cls_feats.view(roi_feats.size(0), -1).contiguous()
        reg_feats = self.reg(self.reg_conv(roi_feats).view(roi_feats.size(0)*roi_feats.size(1), -1).contiguous())
        reg_feats = reg_feats.view(roi_feats.size(0), roi_feats.size(1), -1).contiguous()

        if rois2 is None:
            return cls_feats, reg_feats

        roi_feats2 = self.roipooling(corr_feat, rois2, rois_index2)
        cls_feats2 = self.cls(self.cls_conv(roi_feats2).view(roi_feats2.size(0) * roi_feats2.size(1), -1).contiguous())
        cls_feats2 = cls_feats2.view(roi_feats2.size(0), -1).contiguous()
        reg_feats2 = self.reg(self.reg_conv(roi_feats2).view(roi_feats2.size(0) * roi_feats2.size(1), -1).contiguous())
        reg_feats2 = reg_feats2.view(roi_feats2.size(0), roi_feats2.size(1), -1).contiguous()

        return cls_feats, reg_feats, cls_feats2, reg_feats2


class reidNet(nn.Module):
    def __init__(self, input_dim=256, nclasses=0, featstride=4):
        super(reidNet, self).__init__()
        self.featstride = featstride
        self.roipooling = RoIAlign(36, 16)
        self.reid = nn.Sequential(residual(3, input_dim, input_dim*2),
                                  residual(3, input_dim*2, input_dim*2),
                                  residual(3, input_dim*2, input_dim*2))

        self.num_stripes = 6
        self.local_conv_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        for _ in range(self.num_stripes):
            self.local_conv_list.append(convolution(1, input_dim*2, int(input_dim/4)))
            if nclasses > 0:
                fc = nn.Linear(int(input_dim/4), nclasses)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list.append(fc)
        self.local_conv = convolution(1, input_dim*2, int(input_dim/2))
        self.fc = nn.Linear(int(input_dim/2), nclasses)
        init.normal_(self.fc.weight, std=0.001)
        init.constant_(self.fc.bias, 0)

    def forward(self, ims, rois, rois_index):
        roi_feats = self.roipooling(ims, rois, rois_index)
        feats = self.reid(roi_feats)
        assert feats.size(2) % self.num_stripes == 0
        # local feat
        stripe_h = int(feats.size(2) / self.num_stripes)
        local_feat_list = []
        logits_list = []
        for i in range(self.num_stripes):
            local_feat = F.avg_pool2d(feats[:, :, i*stripe_h: (i+1)*stripe_h, :], (stripe_h, feats.size(-1)))
            # N x C x 1 x 1
            local_feat = self.local_conv_list[i](local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)
            if hasattr(self, 'fc_list'):
                logits_list.append(self.fc_list[i](local_feat))

        # global feat
        global_feat = self.local_conv(F.avg_pool2d(feats, (feats.size(-2), feats.size(-1)))).view(feats.size(0), -1)
        local_feat_list.append(global_feat)
        local_feat = torch.cat(local_feat_list, dim=1)
        if hasattr(self, 'fc_list'):
            logits_list.append(self.fc(global_feat))
            return local_feat, logits_list

        return local_feat


class refineNet(nn.Module):
    '''refine the predictions'''
    def __init__(self):
        super(refineNet, self).__init__()

    def forward(self, x):
        pass


if __name__=='__main__':
    import torch
    s = 257
    ims1 = torch.rand(1, 16, s, s).cuda()
    ims2 = torch.rand(1, 16, s, s).cuda()
    rois1 = torch.FloatTensor([[30/s, 30/s, 80/s, 60/s], [35/s, 35/s, 76/s, 62/s]]).cuda()
    rois2 = torch.FloatTensor([[45/s, 50/s, 120/s, 93/s], [30/s, 30/s, 75/s, 60/s]]).cuda()
    rois_index1 = torch.IntTensor([0, 0]).cuda()
    rois_index2 = torch.IntTensor([0, 0]).cuda()
    tracknet = trackerNet(input_dim=16).cuda()
    reidnet  = reidNet(input_dim=16, nclasses=462).cuda()
    cls_feats, reg_feats, cls_feats2, reg_feats2 = tracknet(ims1, ims2, rois1, rois_index1, rois2, rois_index2)
    print(cls_feats.size())
    print(reg_feats.size())
    print(cls_feats2.size())
    print(reg_feats2.size())
    local_feat, logits_list = reidnet(ims1, rois1, rois_index1)
    print(len(logits_list))
    print(local_feat.size())
    print(logits_list[0].size())