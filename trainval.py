import time
import math
import torch
import logging
import argparse
import numpy as np
import os.path as osp
from os import makedirs

from dataset.mot_dataset import MOT
from model.net import Net
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils.vis_tool import Visualizer
from skimage import transform as SKITransform
from model.utils.net_utils import load_pretrain
from utils.utils import save_checkpoint, clip_gradient, decoding, adjust_learning_rate


np.random.seed(1024)
torch.manual_seed(1314)
torch.cuda.manual_seed(888)

parser = argparse.ArgumentParser(description='Training Siamese tracknet for MOT')
parser.add_argument('--rangex', default=5, type=int, help='select range')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='resume epoch index')
parser.add_argument('-j', default=48, type=int, help='number of data loading workers (default:20)')
parser.add_argument('--grid_size', default=7, type=int, help='grid size for roi align layer')
parser.add_argument('--sample_num', default=6, type=int, help='sample num in each gt (be even value)')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency (default=20)')
parser.add_argument('--gpu', default='0', type=str, help='gpu index')
parser.add_argument('-b', default=4, type=int, help='mini-batch size (default:4)')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--lam', default=10, type=float, help='loss balance parameter')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay(default: 1e-4)')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint (default: None)')
parser.add_argument('--vis_env', default='siam_combine', type=str, help='env name for visdom')
parser.add_argument('--save', default='./work', type=str, help='directory for saving')
parser.add_argument('--log', default='log_combine.txt', type=str, help='log_file')
args = parser.parse_args()
print(args)

logger = logging.getLogger('__name__')
logger.setLevel(logging.DEBUG)

vis = Visualizer(env=args.vis_env)
# ------------------------------------------------------------------------------------------------

gpu = [int(i) for i in args.gpu.split(',')]
torch.cuda.set_device(gpu[0])
gpu_num = len(gpu)
logger.info('there are %d gpus are used' % gpu_num)

filepath = 'dataset/MOT/dataset.json'
train_dataset = MOT(file=filepath, train=True, rangex=args.rangex, max_num=args.sample_num)
val_dataset = MOT(file=filepath, train=False, rangex=args.rangex, max_num=args.sample_num)
train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True,
                          collate_fn=train_dataset.collate_fn, num_workers=args.j,
                          pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.b, shuffle=False,
                        collate_fn=val_dataset.collate_fn, num_workers=args.j,
                        pin_memory=True, drop_last=True)
# --------------------------------------------------------------------------------------------------
# optimizer
model = combinet(grid_size=args.grid_size, nclasses=477)
trk_params = [p for name, p in model.named_parameters() if name.startswith('trk_') or name.startswith('reid_fc')]
trainable_params = [{"params": trk_params, "lr": args.lr},]
optimizer = torch.optim.SGD(trainable_params, lr=1e-4, momentum=args.momentum, weight_decay=args.weight_decay)

# model
model.cuda()
if gpu_num > 1:
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu)

if args.resume:
    if osp.isfile(args.resume):
        logger.info('==>  loading  checkpoint %s', args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        load_pretrain(model, checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info('==> checkpoint %s (epoch %s)', args.resume, args.start_epoch)
    else:
        logger.warning('==> checkpoint %s not found! ', args.resume)

cudnn.benchmark = True
cudnn.deterministic = True

args.save_dir = osp.join(args.save, 'MOT')

# ----------------------------------------------------------------------------------------------------
criterion = ReIDLoss(margin=0.9, sample_num=args.sample_num, bs=args.b)
evaluator = Evaluator(sample_num=args.sample_num, bs=args.b)


def show_img_with_bbox(ims1, ims2, rois1, rois2, pred):
    # show inter results
    rois1 = rois1.data.cpu().numpy()
    rois2 = rois2.data.cpu().numpy()  # xmin, ymin, xmax, ymax
    pred = pred.data.cpu().numpy()
    mean = np.expand_dims(np.expand_dims(np.array([109, 120, 119]),
                                         axis=1), axis=1).astype(np.float32)
    im1 = (ims1.data.cpu().numpy() + mean)   # 3 x H x W
    im2 = (ims2.data.cpu().numpy() + mean)   # 3 x H x W
    vis.image_bbox('img1', im1[[2, 1, 0]], rois1[:, [1, 0, 3, 2]])
    vis.image_bbox('img2', im2[[2, 1, 0]], np.concatenate((rois2, pred), axis=0)[:, [1, 0, 3, 2]], gt=True)

#train
def train(train_loader, model, optimizer, epoch, lr):
    avg = AverageMeter()
    model.train()
    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    for it, dataItems in enumerate(train_loader):
        end = time.time()
        ims = Variable(torch.cat((dataItems[0], dataItems[1]), dim=0)).cuda()
        rois = torch.cat((dataItems[2], dataItems[3]), dim=0)
        idx = torch.FloatTensor(np.concatenate([[i] * args.sample_num for i in range(len(ims))]).astype(np.float32))
        rois_idx = Variable(torch.cat([idx.view(-1, 1), rois], dim=1).contiguous()).cuda()
        id_targets = Variable(torch.cat((dataItems[4], dataItems[5]), dim=0)).cuda()
        reg_targets = Variable(dataItems[6]).cuda()

        reg_pred, logits_list, corr_feat, local_feats_list = model(ims, rois_idx)

        # (reg_pred, soft_feat, corr_feat, reid_feat)
        # loss
        reg_loss = _smooth_l1_loss(reg_pred, reg_targets)
        reid_feat = torch.cat(local_feats_list, dim=1)
        cls_loss, trp_loss = criterion._combine(reid_feat, logits_list, id_targets)

        loss = cls_loss + trp_loss + args.lam * reg_loss
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(model, 10)
        if is_valid_number(loss.data[0]):
            optimizer.step()

        batch_time = time.time() - end
        # eval
        fscore = evaluator.eval_combine(reid_feat, id_targets)
        N = int(len(rois) /2)
        anchor_rois = rois_idx[:N, 1:].contiguous().view(-1, 4).contiguous()
        pred = decoding(anchor_rois, reg_pred)  # Variable, Variable -> Variable
        reg_acc = evaluator.eval_reg(pred,
                                     rois_idx[N:, 1:].contiguous().view(-1, 4).contiguous())

        avg.update(batch_time=batch_time, loss=loss.data[0], reg_loss=reg_loss.data[0],
                   cls_loss=cls_loss.data[0], trp_loss=trp_loss.data[0],
                   reg_acc=reg_acc, cls_acc=fscore)


        if (it + 1) % args.print_freq == 0:
            show_img_with_bbox(ims[0], ims[args.b], rois_idx[:args.sample_num,1:],
                               rois_idx[args.sample_num*args.b:args.sample_num*(args.b+1),1:],
                               pred[:args.sample_num])

            t = corr_feat[0].data.cpu().numpy()
            sample_idx = [0, 8, 16, 17 * 8, 17 * 8 + 8, 17 * 8 + 16, 17 * 16, 17 * 16 + 8, 17 * 16 + 16]
            heatmap = np.zeros((3 * 60, 3 * 80))
            for i in range(3):
                for j in range(3):
                    crop_patch = t[sample_idx[i * 3 + j]]
                    max_abs_value = max(max(abs(np.max(crop_patch)), abs(np.min(crop_patch))), 1e-6)
                    heatmap[i * 60:(i + 1) * 60, j * 80:(j + 1) * 80] = SKITransform.resize(
                        crop_patch / max_abs_value, (60, 80))[::-1]
            vis.img_heatmap('heatmap', heatmap)

            vis.plot_many({'loss': avg.avg('loss'), 'reg_loss': avg.avg('reg_loss'),
                           'cls_loss': avg.avg('cls_loss'), 'reg_acc': avg.avg('reg_acc'),
                           'cls_acc': avg.avg('cls_acc'), 'trp_loss':avg.avg('trp_loss')})
            left_time = (len(train_loader) - it) * batch_time / 3600.  # hour
            log_str = 'Epoch: [{0}][{1}/{2}]  lr: {lr:.6f}  {batch_time:s} \n' \
                      '{loss:s} \t {reg_loss:s} \t {cls_loss:s} \t {trp_loss:s} \n' \
                      '{reg_acc:s} \t {cls_acc:s} \t remaining time:{left_time:.4}h\n'.format(
                        epoch, it + 1, len(train_loader), lr=lr, batch_time=avg.batch_time,
                        loss=avg.loss, reg_loss=avg.reg_loss, cls_loss=avg.cls_loss, trp_loss=avg.trp_loss,
                        reg_acc=avg.reg_acc, cls_acc=avg.cls_acc, left_time=left_time)
            vis.log(log_str, win='train_log')
            logger.info(log_str)


# valid
def validate(val_loader, model):
    avg = AverageMeter()
    model.eval()
    end = time.time()
    logger.info('**' * 10 + ' start validate ' + '**' * 10)
    for it, dataItems in enumerate(val_loader):
        avg.update(data_time=(time.time() - end))
        ims = Variable(torch.cat((dataItems[0], dataItems[1]), dim=0)).cuda()
        rois = torch.cat((dataItems[2], dataItems[3]), dim=0)
        idx = torch.FloatTensor(np.concatenate([[i] * args.sample_num for i in range(len(ims))]).astype(np.float32))
        rois_idx = Variable(torch.cat([idx.view(-1, 1), rois], dim=1).contiguous()).cuda()
        id_targets = Variable(torch.cat((dataItems[4], dataItems[5]), dim=0)).cuda()
        reg_targets = Variable(dataItems[6]).cuda()

        reg_pred, logits_list, corr_feat, local_feats_list = model(ims, rois_idx)

        # (reg_pred, soft_feat, corr_feat, reid_feat)
        # loss
        reg_loss = _smooth_l1_loss(reg_pred, reg_targets)
        reid_feat = torch.cat(local_feats_list, dim=1)
        cls_loss, trp_loss = criterion._combine(reid_feat, logits_list, id_targets)

        loss = cls_loss + trp_loss + args.lam * reg_loss

        # eval
        fscore = evaluator.eval_combine(reid_feat, id_targets)
        N = int(len(rois) / 2)
        anchor_rois = rois_idx[:N, 1:].contiguous().view(-1, 4).contiguous()
        pred = decoding(anchor_rois, reg_pred)  # Variable, Variable -> Variable
        reg_acc = evaluator.eval_reg(pred,
                                     rois_idx[N:, 1:].contiguous().view(-1, 4).contiguous())

        batch_time = time.time() - end
        avg.update(batch_time=batch_time,  reg_acc=reg_acc, cls_acc=fscore, loss=loss.data[0])

        if (it + 1) % args.print_freq == 0:
            vis.plot_many({'val_loss': avg.avg('loss'), 'val_reg_acc': avg.avg('reg_acc'), 'val_cls_acc': avg.avg('cls_acc')})
    vstr = '*'*20 + 'Loss: {:.4f}\t  RegACC : {:.4f} \t ClsACC: {:.4f}'.format(
        avg.avg('loss'), avg.avg('reg_acc'), avg.avg('cls_acc')) + '**' * 10
    vis.log(vstr, win='val_log')
    logger.info(vstr)
    return avg.avg('loss')
# ---------------------------------------------------------------------------------------------------------------

def adj_lr(optimizer, epoch):
    if epoch in [120, 150]:
        adjust_learning_rate(optimizer)
    return optimizer.param_groups[0]['lr']

def main():
    is_best = False
    best_loss = 100
    for epoch in range(args.start_epoch, args.epochs):
        lr = adj_lr(optimizer, epoch)
        train(train_loader, model, optimizer, epoch, lr)
        if (epoch + 1) % 2 == 0 or epoch + 1 == args.epochs:
            # validation
            loss = validate(val_loader, model)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            vis.plot('best_loss', best_loss)
        if not osp.exists(args.save):
            makedirs(args.save)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            },
            is_best,
            os.path.join(args.save_dir, 'curr_checkpoint_comb.pth'),
            os.path.join(args.save_dir, 'best_combine.pth'))


if __name__ == '__main__':
    main()

