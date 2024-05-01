# https://github.com/sovit-123/CamVid-Image-Segmentation-using-FCN-ResNet50-with-PyTorch/blob/master/model.py
import torchvision.models as models
from torchvision.models.segmentation.fcn import FCN
import torch.nn as nn

def Model():
    obj = FCN_ResNet50()
    return obj.model

class FCN_ResNet50:
    def __init__(self, weights=None, requires_grad=True):
        self.weights = weights
        self.model = models.segmentation.fcn_resnet50(weights=self.weights, progress=True)
        
        self.model.requires_grad = requires_grad
        for param in self.model.parameters():
            param.requires_grad = self.model.requires_grad

        self.model.classifier[4] = nn.Conv2d(512, 32, kernel_size=(1,1))
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, 32, kernel_size=(1,1))
        
        self.model.name = 'fcn_resnet50'
