# https://github.com/sovit-123/CamVid-Image-Segmentation-using-FCN-ResNet50-with-PyTorch/blob/master/model.py
import torchvision.models as models
import torch.nn as nn

def Model(weights=None, requires_grad=True):
    model = models.segmentation.fcn_resnet50(weights=weights, progress=True)

    for param in model.parameters():
        param.requires_grad = requires_grad

    model.classifier[4] = nn.Conv2d(512, 32, kernel_size=(1,1))
    
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 32, kernel_size=(1,1))

    return model