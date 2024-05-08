import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://github.com/yassouali/pytorch_segmentation/blob/master/utils/metrics.py
def pix_accuracy(preds, gt, labelled_gt):
    # number of labelled pixels
    pixel_labeled = labelled_gt.to(device).sum()
    # number of correctly predicted pixels
    pixel_correct = ((preds == gt.to(device)) * labelled_gt).sum()
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def iou(preds, gt, labelled_gt, num_classes=32, ret_iou_list=False):
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

    # convert to numpy array
    area_inter = area_inter.cpu().numpy()
    area_union = area_union.cpu().numpy()
    iou_list = None
    if ret_iou_list:
        iou_list = []
        
        area_inter = list(area_inter)
        area_union = list(area_union)

        for i in range(num_classes):
            # append intersection, union for each class label
            iou_list.append((area_inter[i], area_union[i]))
        iou_list = np.array(iou_list)
    
    return area_inter, area_union, iou_list


def eval_metrics(output, gt, num_classes=32, ret_iou_list=False):
    # _, index = predictions class labels
    _, preds = torch.max(output, dim=1)
    preds += 1
    gt += 1

    # filtered ground truths
    labelled_gt = (gt > 0) * (gt <= num_classes)
    correct, labelled = pix_accuracy(preds, gt, labelled_gt.to(device))
    intersection, union, iou_list = iou(preds, gt, labelled_gt.to(device), ret_iou_list=ret_iou_list)

    return np.round(correct, 4), np.round(labelled, 4), np.round(intersection, 4), np.round(union, 4), iou_list

