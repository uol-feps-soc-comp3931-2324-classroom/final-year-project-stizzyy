# https://github.com/sovit-123/CamVid-Image-Segmentation-using-FCN-ResNet50-with-PyTorch/blob/master/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model():
    obj = FCN_ResNet50()
    return obj.model

class FCN_ResNet50:
    def __init__(self, weights=None, requires_grad=True):
        self.weights = weights
        self.model = models.segmentation.fcn_resnet50(weights_backbone=ResNet50_Weights.IMAGENET1K_V2, progress=True)
        
        self.model.requires_grad = requires_grad
        for param in self.model.parameters():
            param.requires_grad = self.model.requires_grad

        self.model.classifier[4] = nn.Conv2d(512, 32, kernel_size=(1,1))
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, 32, kernel_size=(1,1))
        
        self.model.name = 'fcn_resnet50'

fcn_resnet_model = model().to(device)