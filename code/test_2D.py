import argparse
import logging
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
from medpy import metric

torch.cuda.set_device(0)

from dataloaders.dataset import BaseDataSets_2D
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff


from networks.ddnet import DDNet



logging.getLogger().setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_root', default='../model/BrainMRI_96/',  #
                        type=str, help="model path")
    parser.add_argument('--model', '-m', default='ddnet_best_model.pth',
                        metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white", default=0.5)
    parser.add_argument('--scale', '-s', type=float,help="Scale factor for the input images",
                        default=0.5)
    parser.add_argument('--num_classes', type=int, default=1,help='output channel of network')  # 需要修改

    return parser.parse_args()


args = get_args()

def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:#求有向 Hausdorff 距离
    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def haussdorf(preds, target):
    res = numpy_haussdorf(preds, target)
    return res

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def hunxiao(preds, target):

    n_pred = preds.ravel()
    n_target = target.astype('int64').ravel()

    tn, fp, fn, tp = confusion_matrix(n_target, n_pred).ravel()

    smooh = 1e-10
    sensitivity = tp / (tp + fn + smooh)
    specificity = tn / (tn + fp + smooh)
    Accuracy = (tp + tn) / (tn + tp + fp + fn + smooh)
    precision = tp / (tp + fp + smooh)
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity + smooh)

    return sensitivity, specificity, Accuracy, precision, f1_score

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def predict_img(net, full_img, device, scale_factor=0.5, out_threshold=0.5):
    net.eval()
    img = BaseDataSets_2D.preprocess(full_img, scale_factor)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)

    img = img.unsqueeze(0)#增加一个维度
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output, _ = net(img)

        if args.num_classes > 1:
            probs = F.softmax(output[0], dim=1)
        else:
            probs = torch.sigmoid(output[0])

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        res = np.int64(full_mask > out_threshold) ##判断是0是1

    return res


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8)) #实现array到image的转换


if __name__ == "__main__":
    start = time.perf_counter()
    if not os.path.exists('../predict_output/'+ args.model_root.split('/',6)[2] +'_'+ args.model_root.split('/',6)[3]+ '/picture/'):
        os.makedirs('../predict_output/' + args.model_root.split('/',6)[2] +'_'+ args.model_root.split('/',6)[3]+ '/picture/')

    test_img_paths = glob('../data/BrainMRI/test/imgs/*')
    test_mask_paths = glob('../data/BrainMRI/test/masks/*')


    net = DDNet(3, 1)    # 需要修改

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    net.load_state_dict(torch.load(args.model_root + args.model, map_location=device))

    logging.info("Model loaded !")

    sensitivity = []
    specificity = []
    Accuracy = []
    f1_score = []
    iou = []
    hau_d = []
    Jc = []
    Hd95 = []

    for i in tqdm(range(len(test_img_paths))):
        img = Image.open(test_img_paths[i])
        img = img.convert("RGB")
        mask = Image.open(test_mask_paths[i])
        mask = mask.convert("L")
        w, h = mask.size
        newW, newH = int(args.scale * w), int(args.scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = mask.resize((256, 256))
        mask_nd = np.array(pil_img)
        mask_s = mask_nd.astype('float32') / 255

        pd = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        lm, ty, acc, pre, f1 = hunxiao(pd, mask_s)

        jaccard = iou_score(pd,mask_s)
        hd = haussdorf(pd,mask_s)
        result = mask_to_image(pd) #实现array到image的转换
        result.save('../predict_output/'+args.model_root.split('/',6)[2] +'_'+args.model_root.split('/',6)[3]+ '/picture/' + os.path.basename(test_img_paths[i]))

        sensitivity.append(lm)
        specificity.append(ty)
        Accuracy.append(acc)
        f1_score.append(f1)

        iou.append(jaccard)
        hau_d.append(hd)


    print('sensitivity: %.4f' % np.mean(sensitivity))
    print('specificity: %.4f' % np.mean(specificity))
    print('Accuracy: %.4f' % np.mean(Accuracy))
    print('dice(f1_score): %.4f' % np.mean(f1_score))
    print('Jaccard(iou): %.4f' % np.mean(iou))
    print('HD: %.4f' % np.mean(hau_d))

    end = time.perf_counter()
    print('Running time: %s Seconds' % (end - start))
