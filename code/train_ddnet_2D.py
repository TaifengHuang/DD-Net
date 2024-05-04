import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss,BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from glob import glob

from dataloaders.dataset import (BaseDataSets_2D, RandomGenerator_2D, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from utils import val_2D

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def load_net(model, path):
    state = torch.load(str(path))
    model.load_state_dict(state)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

def dice1_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


imgs_dir = glob('../data/brainMRI/train/imgs/*')
masks_dir = glob('../data/brainMRI/train/masks/*')
val_imgs_dir = glob('../data/brainMRI/val/imgs/*')
val_masks_dir = glob('../data/brainMRI/val/masks/*')

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='DD-Net', help='experiment_name')
parser.add_argument('--model', type=str, default='ddnet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--critic_lr', type=float,  default=0.0001, help='DAN learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=1, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=96,
                    help='labeled data')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.25, help='weight to balance all losses')

args = parser.parse_args()

def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    labeled_num = args.labeled_num


    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    def create_emamodel(ema=False):
        model = net_factory(net_type='unet', in_chns=3, class_num=num_classes)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_emamodel(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets_2D(imgs_dir=imgs_dir, masks_dir=masks_dir, split="train", transform=transforms.Compose([
        RandomGenerator_2D(args.patch_size)
    ]))
    db_val = BaseDataSets_2D(imgs_dir=val_imgs_dir, masks_dir=val_masks_dir, split="val")

    n_train = len(db_train) # 566
    n_val = len(db_val) # 242
    print("Total train num is: {}, labeled num is: {}".format(
        n_train, labeled_num))

    labeled_idxs = list(range(0, labeled_num))
    unlabeled_idxs = list(range(labeled_num, n_train))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = BCELoss()

    mse_criterion = losses.mse_loss

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    cur_threshold = 1 / 2

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]

            outputs, deep_out2 = model(volume_batch)

            with torch.no_grad():
                ema_out = ema_model(unlabeled_volume_batch)  #bs/2
                ema_out_soft = torch.sigmoid(ema_out)

            loss_seg_dice = 0

            output1_soft = torch.sigmoid(outputs[0])  # [b,4,256,256]
            output2_soft = torch.sigmoid(outputs[1])  # [b,4,256,256]

            deep_out2_soft1 = torch.sigmoid(deep_out2[0])
            deep_out2_soft2 = torch.sigmoid(deep_out2[1])
            deep_out2_soft3 = torch.sigmoid(deep_out2[2])
            deep_out2_soft4 = torch.sigmoid(deep_out2[3])

            mask_high = (output1_soft >= 0.75) & (output2_soft >= 0.75)  # [b,4,256,256]
            mask_mid = ((output1_soft > 0.75) & (output2_soft < 0.75)) | ((output1_soft < 0.75) & (output2_soft > 0.75))
            mask_low = (output1_soft < 0.75) & (output2_soft < 0.75)

            high_output1 = torch.mul(mask_high, outputs[0])  # [b,4,256,256]
            high_output2 = torch.mul(mask_high, outputs[1])  # [b,4,256,256]
            high_output1_soft = torch.mul(mask_high, output1_soft)  # [b,4,256,256]
            high_output2_soft = torch.mul(mask_high, output2_soft)  # [b,4,256,256]

            mid_output1 = torch.mul(mask_mid, outputs[0])
            mid_output2 = torch.mul(mask_mid, outputs[1])

            low_output1 = torch.mul(mask_low, outputs[0])
            low_output2 = torch.mul(mask_low, outputs[1])

            loss_seg_dice += dice1_loss(output1_soft[:labeled_bs, ...], label_batch[:labeled_bs])  # [b,4,256,256]和[b,1,256,256]做损失
            loss_seg_dice += dice1_loss(output2_soft[:labeled_bs, ...], label_batch[:labeled_bs])

            loss_deep_sup2_1 = dice1_loss(deep_out2_soft1[:labeled_bs, ...], label_batch[:labeled_bs])
            loss_deep_sup2_2 = dice1_loss(deep_out2_soft2[:labeled_bs, ...], label_batch[:labeled_bs])
            loss_deep_sup2_3 = dice1_loss(deep_out2_soft3[:labeled_bs, ...], label_batch[:labeled_bs])
            loss_deep_sup2_4 = dice1_loss(deep_out2_soft4[:labeled_bs, ...], label_batch[:labeled_bs])

            loss_deep_sup = (loss_deep_sup2_1+loss_deep_sup2_2+loss_deep_sup2_3+loss_deep_sup2_4)/4   # 转置卷积

            pseudo_high_output1 = torch.argmax(high_output1_soft[labeled_bs:], dim=1)
            pseudo_high_output2 = torch.argmax(high_output2_soft[labeled_bs:], dim=1)

            pseudo_supervision = 0

            pseudo_supervision += ce_loss(torch.sigmoid(high_output1[labeled_bs:]), pseudo_high_output2.unsqueeze(1).float().detach())
            pseudo_supervision += ce_loss(torch.sigmoid(high_output2[labeled_bs:]), pseudo_high_output1.unsqueeze(1).float().detach())

            loss_mm = dice1_loss(torch.sigmoid(mid_output1[labeled_bs:] / 2), torch.sigmoid(mid_output2[labeled_bs:] / 2))

            low_loss = mse_criterion(torch.sigmoid(low_output1[labeled_bs:] / 0.5), torch.sigmoid(low_output2[labeled_bs:] / 0.5))

            pseudo = torch.argmax(ema_out_soft, dim=1)
            loss_u = ce_loss((output1_soft[labeled_bs:]+output2_soft[labeled_bs:])*0.5, pseudo.unsqueeze(1).float().detach())

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            supervised_loss = loss_seg_dice

            loss = supervised_loss + loss_deep_sup + consistency_weight * pseudo_supervision + (1 - consistency_weight) * loss_mm + 2 * low_loss + consistency_weight * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1
            logging.info('iteration %d :loss : %03f,  loss_seg_dice: %03f,  loss_kd: %03f, pseudo_supervision: %03f, low_loss: %03f,loss3: %03f'
                         % (iter_num, loss, loss_seg_dice, loss_mm, pseudo_supervision, low_loss,loss_u ))

            if iter_num > 0 and iter_num % 20 == 0:
                model.eval()
                dice_sum, hs_sum = val_2D.eval_modal(model, valloader, device, n_val, classes=num_classes)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                                      dice_sum, iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                                      hs_sum, iter_num)

                writer.add_scalar('info/model1_val_dice', dice_sum, iter_num)
                writer.add_scalar('info/model1_val_hd', hs_sum, iter_num)

                if dice_sum > best_performance:
                    best_performance = dice_sum
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_performance))    # round(best_performance, 4)
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, dice_sum, hs_sum))
                model.train()

            if iter_num % 15000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_snapshot_path = "../model/{}_{}_labeled/{}/pre_train".format(
        args.exp, args.labeled_num, args.model)
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)