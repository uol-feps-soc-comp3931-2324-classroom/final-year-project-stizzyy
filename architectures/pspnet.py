import torch
import torch.nn as nn
import torch.nn.functional as F

from psp_backbone import resnet50

# https://github.com/hszhao/semseg/blob/master/model/pspnet.py
class PSPM(nn.Module):
    def __init__(self, in_dimensions, out_dimensions, bins):
        super(PSPM, self).__init__()
        self.features = nn.ModuleList(
            [nn.Sequential(
                nn.AdaptiveMaxPool2d(bin),
                nn.Conv2d(in_dimensions, out_dimensions, kernel_size=1),
                nn.BatchNorm2d(out_dimensions),
                nn.ReLU(inplace=True)
            ) for bin in bins
            ]
        )

    def forward(self, x):
        out = []
        out.append(x)

        x_size = x.size()[2:] # h, w
        for feature in self.features:
            # upsample low-res feature maps to original feature map size by bilinear interpolation
            out.append(F.interpolate(feature(x), x_size, mode='bilinear', align_corners=True))

        return torch.cat(out, dim=1)
    


class PSPNet(nn.Module):
    name = 'pspnet'

    def __init__(self, layers=50, bins=(1, 2, 3, 6), num_classes=32, use_ppm=True, zoom_factor=2, pretrained=False):
        super(PSPNet, self).__init__()
        self.use_ppm = use_ppm
        self.zoom_factor = zoom_factor
        self.dropout = 0.1

        if layers == 50: # so far only support resnet50
           backbone =  resnet50(pretrained=pretrained)
        
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
            self.ppm = PSPM(in_dimensions=feature_dims, out_dimensions=int(feature_dims/len(bins)), bins=bins)
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
        x_size = x.size()[2:]
        h = int(((x_size[0] - 1) / 8) * self.zoom_factor + 1)
        w = int(((x_size[1] - 1) / 8) * self.zoom_factor + 1)

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

        # resizing based on zoom factor
        # HIGHER zoom factor = preserves spatial detail
        # LOWER zoom factor = more computationally efficient
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux_clf(x_aux)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)  
            return x, aux
        else:
            return x
            

model = PSPNet(layers=50, bins=(1, 2, 3, 6), zoom_factor=2, use_ppm=True).cuda()



