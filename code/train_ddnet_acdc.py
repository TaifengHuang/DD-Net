"""
    创建时间：2023/11/12 13:50
"""
import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss,KLDivLoss,BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, ramps, val
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DD-Net', help='experiment_name')
parser.add_argument('--model', type=str, default='ddnet', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3, help='labeled data')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance all losses')
parser.add_argument('--beta', type=float,  default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--temp', default=1, type=float)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
def patients_to_slices(dataset, patiens_num):   # _,7
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 34, "3": 68, "5": 92, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

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

def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    def create_emamodel(ema=False):
        model = net_factory(net_type='unet', in_chns=1, class_num=num_classes)
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

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)

    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()

    mse_criterion = losses.mse_loss

    dice_loss = losses.DiceLoss(n_classes=num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]

            outputs, deep_out2 = model(volume_batch)

            with torch.no_grad():
                ema_out = ema_model(unlabeled_volume_batch)  #bs/2
                ema_out_soft = F.softmax(ema_out, dim=1)

            loss_seg_dice = 0

            output1_soft = F.softmax(outputs[0], dim=1)    # [b,4,256,256]
            output2_soft = F.softmax(outputs[1], dim=1)    # [b,4,256,256]


            deep_out2_soft1 = F.softmax(deep_out2[0], dim=1)
            deep_out2_soft2 = F.softmax(deep_out2[1], dim=1)
            deep_out2_soft3 = F.softmax(deep_out2[2], dim=1)
            deep_out2_soft4 = F.softmax(deep_out2[3], dim=1)


            mask_high = (output1_soft >= 0.80) & (output2_soft >= 0.80) # [b,4,256,256]
            mask_mid = ((output1_soft > 0.80) & (output2_soft < 0.80)) | ((output1_soft < 0.80) & (output2_soft > 0.80))
            mask_low = (output1_soft < 0.80) & (output2_soft < 0.80)

            high_output1 = torch.mul(mask_high, outputs[0])    # [b,4,256,256]
            high_output2 = torch.mul(mask_high, outputs[1])    # [b,4,256,256]
            high_output1_soft = torch.mul(mask_high, output1_soft)  # [b,4,256,256]
            high_output2_soft = torch.mul(mask_high, output2_soft)  # [b,4,256,256]

            mid_output1 = torch.mul(mask_mid, outputs[0])
            mid_output2 = torch.mul(mask_mid, outputs[1])

            low_output1 = torch.mul(mask_low, outputs[0])
            low_output2 = torch.mul(mask_low, outputs[1])

            loss_seg_dice += dice_loss(output1_soft[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))
            loss_seg_dice += dice_loss(output2_soft[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))

            loss_deep_sup2_1 = dice_loss(deep_out2_soft1[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))
            loss_deep_sup2_2 = dice_loss(deep_out2_soft2[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))
            loss_deep_sup2_3 = dice_loss(deep_out2_soft3[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))
            loss_deep_sup2_4 = dice_loss(deep_out2_soft4[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))

            loss_deep_sup = (loss_deep_sup2_1+loss_deep_sup2_2+loss_deep_sup2_3+loss_deep_sup2_4)/4   # 转置卷积

            pseudo_high_output1 = torch.argmax(high_output1_soft[labeled_bs:], dim=1)
            pseudo_high_output2 = torch.argmax(high_output2_soft[labeled_bs:], dim=1)

            pseudo_supervision = 0

            pseudo_supervision += ce_loss(high_output1[labeled_bs:], pseudo_high_output2.long().detach())
            pseudo_supervision += ce_loss(high_output2[labeled_bs:], pseudo_high_output1.long().detach())

            loss_mm = dice1_loss(F.softmax(mid_output1[labeled_bs:] / 2, dim=1), F.softmax(mid_output2[labeled_bs:] / 2, dim=1))


            low_loss = mse_criterion(F.softmax(low_output1[labeled_bs:] / 0.5, dim=1), F.softmax(low_output2[labeled_bs:] / 0.5, dim=1))

            pseudo = torch.argmax(ema_out_soft, dim=1)

            loss_u = ce_loss((outputs[0][labeled_bs:]+outputs[1][labeled_bs:])*0.5, pseudo.long().detach())




            consistency_weight = get_current_consistency_weight(iter_num // 150)
            supervised_loss = loss_seg_dice


            loss = supervised_loss + loss_deep_sup + consistency_weight * pseudo_supervision + (1 - consistency_weight)* loss_mm + 2 *low_loss + consistency_weight * loss_u


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            iter_num = iter_num + 1
            logging.info('iteration %d : loss : %03f,  supervised_loss: %03f,loss_deep_sup: %03f, pseudo_supervision: %03f, loss_kd: %03f, low_loss: %03f,loss3: %03f'
                          % (
            iter_num, loss, supervised_loss,loss_deep_sup, pseudo_supervision,  loss_mm,low_loss,loss_u))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                output = torch.argmax(torch.softmax(outputs[0], dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/ACDC_{}_{}_{}_labeled".format(args.model, args.exp, args.labeled_num)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('../code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # torch.backends.cudnn.enabled = False

    train(args, snapshot_path)

