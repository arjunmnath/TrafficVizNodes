import torch
import torch.nn as nn
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .backbones.resnext_ibn import resnext101_ibn_a
from .layers.pooling import GeM, GeneralizedMeanPoolingP


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_name = cfg.MODEL.NAME
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = model_name

        if model_name == "resnet50_ibn_a":
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print("using resnet50_ibn_a as a backbone")
        elif model_name == "resnet101_ibn_a":
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print("using resnet101_ibn_a as a backbone")
        elif model_name == "resnext101_ibn_a":
            self.in_planes = 2048
            self.base = resnext101_ibn_a()
            print("using resnext101_ibn_a as a backbone")
        else:
            raise ValueError(f"Unsupported backbone: {model_name}")

        if cfg.MODEL.POOLING_METHOD == "gempoolP":
            print("using GeMP pooling")
            self.gap = GeneralizedMeanPoolingP()
        elif cfg.MODEL.POOLING_METHOD == "gempool":
            print("using GeM pooling")
            self.gap = GeM(freeze_p=False)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)

    def forward(self, x, label=None, cam_label=None):
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(-1, self.in_planes)  # flatten to (bs, 2048)

        if self.neck == "no":
            feat = global_feat
        elif self.neck == "bnneck":
            feat = self.bottleneck(global_feat)

        if self.neck_feat == "after":
            return feat
        else:
            return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location="cpu")
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]

        # Load weights, ignoring classification heads/loss parameters if they exist
        state_dict = self.state_dict()
        for i in param_dict:
            if "classifier" in i or "arcface" in i:
                continue
            key = i.replace("module.", "")
            if key in state_dict:
                state_dict[key].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))


def make_model(cfg, num_class, camera_num=0, view_num=0):
    model = Backbone(num_class, cfg)
    return model
