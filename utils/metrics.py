import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://github.com/yassouali/pytorch_segmentation/blob/master/utils/metrics.py
def iou(preds, gt, labelled_gt, num_classes=32):
    # filter out invalid pixels
    preds = preds * labelled_gt.to(device).long()
    # compute intersection. if match then 1, otherwise 0
    inter = preds * (preds == gt.to(device)).long()

    # compute histograms. distribution of pixels per class
    area_preds = torch.histc(preds.float(), bins=num_classes, min=1, max=num_classes)
    area_inter = torch.histc(inter.float(), bins=num_classes, min=1, max=num_classes)
    area_gt = torch.histc(gt.float(), bins=num_classes, min=1, max=num_classes)

    # compute union
    area_union = area_preds + area_gt - area_inter

    return area_inter.cpu().numpy(), area_union.cpu().numpy()


def eval_metrics(output, gt, num_classes=32):
    # _, index = predictions class labels
    _, preds = torch.max(output, dim=1)
    preds += 1
    gt += 1

    # filtered ground truths
    labelled_gt = (gt > 0) * (gt <= num_classes)
    intersection, union = iou(preds, gt, labelled_gt.to(device))

    return np.round(intersection, 4), np.round(union, 4)

