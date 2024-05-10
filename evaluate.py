import torch
import torch.nn as nn
import numpy as np
import cv2
from albumentations import Compose, Resize, Normalize
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

from models.pspnet import *
from camvid import test_dataloader, test_dataset
from utils.helpers import draw_test_seg_map
from utils.metrics import eval_metrics, load_metrics_from_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_model(model, dataset, dataloader, path='gen'):
    criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)

    loaded_dict = torch.load(os.path.join(path, model.name, 'e_FINAL.pth'))
    model.load_state_dict(loaded_dict['model_state_dict'])
    model.eval().to(device)

    acc_loss = 0
    acc_correct, acc_labelled = 0, 0
    acc_inter, acc_union = 0, 0

    acc_iou_list = np.zeros((32, 2), dtype=np.float32)

    with torch.no_grad():
        n_iterations = int(len(dataset)/dataloader.batch_size)
        progress_bar = tqdm(dataloader, total=n_iterations)

        batch_counter = 0

        for i, (data, label) in enumerate(progress_bar):
            batch_counter += 1

            data, label = data.to(device), label.to(device)

            out = model(data)
            
            loss = criterion(out, label.long())
            acc_loss += loss.item()

            # segmentation mask
            if i == n_iterations - 1:
                draw_test_seg_map(out, data, label, model, path, batch=0)
                
            # EVALUATE METRICS
            c, l, i, u, iou_list = eval_metrics(out.data, label.clone().detach(), ret_iou_list=True)
            i, u = torch.from_numpy(i).to(device), torch.from_numpy(u).to(device)

            # accummulate pixel accuracy
            acc_correct += c
            acc_labelled += l

            # accummulate IoU
            acc_inter += i
            acc_union += u

            # accummulate IoU list
            acc_iou_list += iou_list

            progress_bar.set_description(f'TESTING {model.name}--\t loss : {loss:.3f} | pix_acc : {(c/l):.3f} | miou : {(1.0 * i/(np.spacing(1) + u)).mean():.3f}')

    # average loss
    loss = acc_loss / batch_counter

    # pixel accuracy
    pix_acc = 1.0 * acc_correct / (np.spacing(1) + acc_labelled)

    # miou
    iou = 1.0 * acc_inter / (np.spacing(1) +  acc_union)
    miou = iou.mean().cpu().numpy().min()

    # save iou list as .csv file
    iou_list = np.array(acc_iou_list).reshape(-1, 32)
    df = pd.DataFrame(iou_list, index=['intersection', 'union'])
    df.to_csv(os.path.join(path, model.name, 'iou_list.csv'))

    return loss, pix_acc, miou
        

def test_models(models):
    
    metrics = {}

    for model in models:
        loss, pix_acc, miou = test_model(model, test_dataset, test_dataloader, path='psp_variants')
        metrics[model.name] = [loss, pix_acc, miou]
    
   
    df = pd.DataFrame(metrics)
    print(df)

def get_trainval_metrics(models, path='psp_variants'):
    metrics_dict = {}

    for model in models:
        metrics = load_metrics_from_file(os.path.join(path, model.name))
        
        loss, pix_acc, miou = metrics[0], metrics[1], metrics[2]

        t_metrics, v_metrics = [], []

        for metric in [loss, pix_acc, miou]:
            t_metrics.append(metric['train'][-1])
            v_metrics.append(metric['val'][-1])
       

        metrics_dict[f'{model.name.removeprefix("pspnet_")}_train'] = t_metrics
        metrics_dict[f'{model.name.removeprefix("pspnet_")}_val'] = v_metrics

    df = pd.DataFrame(metrics_dict)
    df.to_csv(os.path.join(path, 'trainval_metrics.csv'))

if __name__ == '__main__':
    models = [base, psp_notpretrained, psp_b1_avg, psp_b1_max, psp_b1236_avg, psp_b1236_max]

    test_models(models)
    get_trainval_metrics(models)