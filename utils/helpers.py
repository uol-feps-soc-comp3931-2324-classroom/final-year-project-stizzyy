import pandas as pd
import numpy as np
import os

DATASET_PATH = 'datasets/CamVid'

def make_class_dict(path = DATASET_PATH):
    df = pd.read_csv(os.path.join(path, 'class_dict.csv'))
    dict = df.set_index('name').T.to_dict('list')

    # convert rgb list -> tuple
    for name, rgb in dict.items():
        dict[name] = tuple(rgb)
    return dict

CLASS_DICT = make_class_dict()


def get_labelled_mask(mask, dict : dict):
    lab_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for class_id, rgb in enumerate(dict.values()):
        rgb = np.array(rgb)
        lab_mask[np.where(np.all(mask == rgb, axis=-1))[:2]] = class_id
    
    lab_mask = lab_mask.astype(int)
    return lab_mask


def make_pth_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)