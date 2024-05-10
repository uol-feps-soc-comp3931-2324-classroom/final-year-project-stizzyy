from torchsummary import summary

from models.pspnet import *

models = [base, psp_notpretrained, psp_b1_avg, psp_b1_max, psp_b1236_avg, psp_b1236_max]


summary(models[4].eval(), (3, 224, 224)) 
