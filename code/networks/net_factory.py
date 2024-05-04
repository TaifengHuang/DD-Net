from networks.unet import UNet
from networks.ddnet import DDNet


def net_factory(net_type="unet", in_chns=1, class_num=4, normalization='batchnorm', has_dropout=True):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "ddnet":
        net = DDNet(in_chns=in_chns, class_num=class_num)
    else:
        net = None
    return net


