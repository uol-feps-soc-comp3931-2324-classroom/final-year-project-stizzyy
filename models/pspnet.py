import torch
import torch.nn as nn
import torch.nn.functional as F

from models.psp_backbone import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://github.com/hszhao/semseg/blob/master/model/pspnet.py


def upsample(input, size):
    out = F.interpolate(input, size=size, mode='bilinear', align_corners=True)
    return out

# PYRAMID PARSING MODULE
class PSPM(nn.Module):
    def __init__(self, in_dimensions, bins=(1, 2, 3, 6)):
        super(PSPM, self).__init__()
        out_dimensions = in_dimensions // len(bins)

        self.feat1_2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(bins[0]),
            nn.Conv2d(in_dimensions, out_dimensions, kernel_size=1)
        )

        self.feat1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(bins[0]),
            nn.Conv2d(in_dimensions, out_dimensions, kernel_size=1),
            nn.BatchNorm2d(out_dimensions, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

        self.feat2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(bins[1]),
            nn.Conv2d(in_dimensions, out_dimensions, kernel_size=1),
            nn.BatchNorm2d(out_dimensions),
            nn.ReLU(inplace=True)
        )

        self.feat3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(bins[2]),
            nn.Conv2d(in_dimensions, out_dimensions, kernel_size=1),
            nn.BatchNorm2d(out_dimensions),
            nn.ReLU(inplace=True)
        )

        self.feat4 = nn.Sequential(
            nn.AdaptiveMaxPool2d(bins[3]),
            nn.Conv2d(in_dimensions, out_dimensions, kernel_size=1),
            nn.BatchNorm2d(out_dimensions),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()[-2:]

        if x.size()[0] != 1:
            out1 = upsample(self.feat1(x), size=x_size)
        else:
            out1 = upsample(self.feat1_2(x), size=x_size)
        out2 = upsample(self.feat2(x), size=x_size)
        out3 = upsample(self.feat3(x), size=x_size)
        out4 = upsample(self.feat4(x), size=x_size)
        
        return torch.cat([x, out1, out2, out3, out4], dim=1)
    


class PSPNet(nn.Module):
    name = 'pspnet'

    def __init__(self, layers=50, bins=(1, 2, 3, 6), num_classes=32, use_deep_backbone=True, use_ppm=True, resize=(224, 224), pretrained=True):
        super(PSPNet, self).__init__()
        self.use_deep_backbone = use_deep_backbone
        self.use_ppm = use_ppm
        self.resize = resize
        self.dropout = 0.1
        self.criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()

        if layers == 50: # so far only support resnet50
           backbone =  resnet50(pretrained=pretrained, use_deep_backbone=self.use_deep_backbone)
        
        if not self.use_deep_backbone:
            self.layer0 = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
            )
        else:
            self.layer0 = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu,
                backbone.conv2, backbone.bn2, backbone.relu,
                backbone.conv3, backbone.bn3, backbone.relu,
                backbone.maxpool
            )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        feature_dims = 2048
        if use_ppm:
            self.ppm = PSPM(in_dimensions=feature_dims, bins=bins)
            feature_dims *= 2

        self.clf = nn.Sequential(
            nn.Conv2d(feature_dims, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        if self.training:
            self.aux_clf = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=self.dropout),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )

    def forward(self, x, y=None):

        # backbone
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        # PPM
        if self.use_ppm:
            x = self.ppm(x)
        
        # classifier
        x = self.clf(x)

        # resizing to 224x224 by default
        x = F.interpolate(x, size=self.resize, mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux_clf(x_aux)
            aux = F.interpolate(aux, size=self.resize, mode='bilinear', align_corners=True)  
            x_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x, x_loss, aux_loss
        else:
            return x
            

psp_model = PSPNet(layers=50, bins=(1, 2, 3, 6), use_deep_backbone=False, use_ppm=True, resize=(224, 224), pretrained=True).to(device)
