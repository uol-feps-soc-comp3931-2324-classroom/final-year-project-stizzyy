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
    def __init__(self, in_dimensions, bins=(1, 2, 3, 6), pool=nn.AdaptiveAvgPool2d):
        super(PSPM, self).__init__()
        out_dimensions = in_dimensions // len(bins)

        self.pool = pool

        self.features = [nn.Sequential(
            self.pool(bin),
            nn.Conv2d(in_dimensions, out_dimensions, kernel_size=1),
            nn.BatchNorm2d(out_dimensions),
            nn.ReLU(inplace=True)
        ) for bin in bins]
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()[-2:]

        out = [x]
        for feature in self.features:
            out.append(upsample(feature(x), size=x_size))
        
        return torch.cat(out, dim=1)
    


class PSPNet(nn.Module):
    name = 'pspnet'

    def __init__(self, layers=50, bins=(1, 2, 3, 6), num_classes=32, use_deep_backbone=False, use_ppm=True, resize=(224, 224), pool=nn.AdaptiveMaxPool2d, pretrained=True):
        super(PSPNet, self).__init__()
        if not use_ppm: 
            self.name = f'{PSPNet.name}_noppm'
        elif not pretrained:
            self.name = f'{PSPNet.name}_notpt'
        else:
            self.name = f'{PSPNet.name}_b{"".join(str(bins))}_{"avg" if isinstance(pool, nn.AdaptiveAvgPool2d) else "max"}'

        self.use_deep_backbone = use_deep_backbone
        self.use_ppm = use_ppm
        self.resize = resize
        self.criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)

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
            self.ppm = PSPM(in_dimensions=feature_dims, bins=bins, pool=pool)
            feature_dims *= 2

        self.clf = nn.Sequential(
            nn.Conv2d(feature_dims, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        if self.training:
            self.aux_clf = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
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
            
# PSP models
psp_noppm = PSPNet(layers=50, use_ppm=False).to(device)
psp_b1_avg = PSPNet(layers=50, bins=(1,), pool=nn.AdaptiveAvgPool2d).to(device)
psp_b1_max = PSPNet(layers=50, bins=(1,), pool=nn.AdaptiveMaxPool2d).to(device)
psp_b1236_avg = PSPNet(layers=50, bins=(1,2,3,6), pool=nn.AdaptiveAvgPool2d).to(device)
psp_b1236_max = PSPNet(layers=50, bins=(1,2,3,6), pool=nn.AdaptiveMaxPool2d).to(device)
psp_notpretrained = PSPNet(layers=50, pretrained=False).to(device)
