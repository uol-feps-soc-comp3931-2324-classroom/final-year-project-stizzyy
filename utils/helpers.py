import torch
from torchvision.utils import draw_segmentation_masks
import pandas as pd
import numpy as np
import os
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = 'datasets/CamVid'

def make_class_dict(path = DATASET_PATH):
    df = pd.read_csv(os.path.join(path, 'class_dict.csv'))
    dict = df.set_index('name').T.to_dict('list')

    # convert rgb list -> tuple
    for name, rgb in dict.items():
        dict[name] = tuple(rgb)
    return dict

CLASS_DICT = make_class_dict()


# https://github.com/sovit-123/CamVid-Image-Segmentation-using-FCN-ResNet50-with-PyTorch/blob/master/utils/helpers.py#L74
def get_labelled_mask(mask):
    lab_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for class_id, rgb in enumerate(CLASS_DICT.values()):
        rgb = np.array(rgb)
        lab_mask[np.where(np.all(mask == rgb, axis=-1))[:2]] = class_id
    
    lab_mask = lab_mask.astype(int)
    return lab_mask


# https://github.com/sovit-123/CamVid-Image-Segmentation-using-FCN-ResNet50-with-PyTorch/blob/master/utils/helpers.py#L89
def draw_seg_map(input, gt, output, epoch, path):
    # render the map every 5 epochs
    if epoch % 5 != 0:
        return
    
    # original image is derived from input
    # prediction segmentation mask derived from output
    num_classes = output[0].shape[0]
    # set alpha, beta, gamma for image blending
    a = 0.6
    b = 1 - a
    y = 0
        
    # COMPUTE SEG MAP
    seg_map = output[0] # first batch
    # find the index(class) with highest value on 0th dim
    # each pixel labelled with highest probable class
    seg_map = torch.argmax(seg_map.squeeze(), dim=0).cpu().numpy()

    # COMPUTE IMAGE
    image = input[0] # first batch
    image = image.cpu().numpy()
    # untransform image
    image = np.transpose(image, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = image.astype(dtype=np.float32)
    # rescale colour: 0-1 -> 0-255
    image = image * 255

    r_mask = np.zeros_like(seg_map, dtype=np.uint8)
    g_mask = np.zeros_like(seg_map, dtype=np.uint8)
    b_mask = np.zeros_like(seg_map, dtype=np.uint8)

    # range() returns list of indices -> identical to labelled class mask
    for label in range(num_classes):
        # create a boolean mask filtering the segmentation map with the class label
        bool_mask = seg_map == label

        # filter channel masks
        # label retrieves the rgb tuple in the values from class dictionary
        # 0 = red, 1 = green, 2 = blue
        # assigns label channel value to valid elements of channel mask
        r_mask[bool_mask] = np.array(list(CLASS_DICT.values()))[label, 0]
        g_mask[bool_mask] = np.array(list(CLASS_DICT.values()))[label, 1]
        b_mask[bool_mask] = np.array(list(CLASS_DICT.values()))[label, 2]

    rgb_mask = np.array(np.stack([r_mask, g_mask, b_mask], axis=2), dtype=np.float32)
    # CV only accepts BGR format
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # INITIALIZATION
    if epoch == 0:
        # ORIGINAL IMAGE
        cv2.imwrite(os.path.join(path, f'_original.jpg'), image)

        # GROUND TRUTH
        gt = gt[0] # first batch
        gt = gt.cpu().numpy()

        gt_r = np.zeros_like(gt, dtype=np.uint8)
        gt_g = np.zeros_like(gt, dtype=np.uint8)
        gt_b = np.zeros_like(gt, dtype=np.uint8)
        
        # range() returns list of indices -> identical to labelled class mask
        for label in range(num_classes):
            # create a boolean mask filtering the ground truth with the class label
            gt_bool_mask = gt == label

            # filter channel masks
            # label retrieves the rgb tuple in the values from class dictionary
            # 0 = red, 1 = green, 2 = blue
            # assigns label channel value to valid elements of channel mask
            gt_r[gt_bool_mask] = np.array(list(CLASS_DICT.values()))[label, 0]
            gt_g[gt_bool_mask] = np.array(list(CLASS_DICT.values()))[label, 1]
            gt_b[gt_bool_mask] = np.array(list(CLASS_DICT.values()))[label, 2]

        gt = np.array(np.stack([gt_r, gt_g, gt_b], axis=2), dtype=np.float32)
        gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(path, f'_gt.jpg'), gt)
        cv2.addWeighted(gt, a, image, b, y, gt)
        cv2.imwrite(os.path.join(path, f'_combined.jpg'), gt)

    # linear blend operator
    cv2.addWeighted(rgb_mask, a, image, b, y, image)
    cv2.imwrite(os.path.join(path, f'smap_e{epoch}.jpg'), image)
