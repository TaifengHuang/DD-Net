import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
import torch.nn as nn


def one_hot_encoder(input_tensor,n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:#求有向 Hausdorff 距离
    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def haussdorf(preds, target):

    n_pred = preds.cpu().numpy().squeeze(0)
    n_target = target.cpu().numpy().squeeze(0)
    res = numpy_haussdorf(n_pred, n_target)
    return res

def dice_coeff(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return loss

def eval_modal(model, vallader, device, nval, classes):  # 三通道
    totdice = 0.0
    toths = 0.0
    model.eval()
    with tqdm(total=nval, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in vallader:
            imgs = batch['image']
            label = batch['label']
            print(imgs.shape,label.shape)
            imgs = imgs.permute(0, 3, 1, 2)
            label = label.permute(0, 3, 1, 2)
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if classes == 1 else torch.long
            label = label.to(device=device, dtype=mask_type)
            img_pred, _ = model(imgs)
            if classes > 1:
                label = one_hot_encoder(label, classes)
                img_pred = torch.softmax(img_pred[0],dim=1)
                hs = 0.0
                dice = 0.0
                for channel in range(img_pred.shape[1]):
                    pred = (img_pred[:, channel, ...] > 0.5).float()
                    hs += haussdorf(pred, label[:, channel, ...])
                    dice += dice_coeff(pred, label[:, channel, ...])
                toths += hs / img_pred.shape[1]
                totdice += dice / img_pred.shape[1]
                pbar.update(imgs.shape[0])
            else:
                img_pred = torch.sigmoid(img_pred[0])
                for true_mask, pred in zip(label, img_pred):
                    pred = (pred > 0.5).float()
                    toths += haussdorf(pred, true_mask.squeeze(dim=1))
                    totdice += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                pbar.update(imgs.shape[0])
    hs_sum = toths / nval
    dice_sum = totdice / nval

    return dice_sum, hs_sum
