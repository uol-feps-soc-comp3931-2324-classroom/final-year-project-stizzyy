import torch
import torch.nn as nn
import numpy as np
import cv2
from albumentations import Compose, Resize, Normalize
from tqdm import tqdm
import os
import pandas as pd

from PIL import Image
from models.fcn_resnet50 import fcn_resnet_model
from models.pspnet import psp_b1_max, PSPNet
from camvid import test_dataloader, test_dataset
from utils.metrics import eval_metrics, load_metrics_from_file, store_metrics_from_list
from utils.visualization import ioulist_to_csv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_model(model, dataset, dataloader, path='gen'):
    is_fcn = True
    criterion = nn.CrossEntropyLoss().to(device)

    if isinstance(model, PSPNet):
        is_fcn = False
        criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)

    loaded_dict = torch.load(os.path.join(path, f'{model.name}/e_FINAL.pth'))
    model.load_state_dict(loaded_dict['model_state_dict'])
    model.eval().to(device)

    acc_loss = 0
    acc_correct, acc_labelled = 0, 0
    acc_inter, acc_union = 0, 0

    acc_iou_list = np.zeros((32, 2), dtype=np.float32)

    with torch.no_grad():
        n_iterations = int(len(dataset)/dataloader.batch_size)
        progress_bar = tqdm(dataloader, total=n_iterations)

        for i, (data, label) in enumerate(progress_bar):
            data, label = data.to(device), label.to(device)


            out = model(data)
            if is_fcn:
                out = out['out']
            
            loss = criterion(out, label.long())

            acc_loss += loss.item()

            c, l, i, u, iou_list = eval_metrics(out, label.clone().detach(), ret_iou_list=True)
            i, u = torch.from_numpy(i).to(device), torch.from_numpy(u).to(device)

            # accummulate pixel accuracy
            acc_correct += c
            acc_labelled += l

            # accummulate IoU
            acc_inter += i
            acc_union += u

            # accummulate IoU list
            acc_iou_list += iou_list

    # average loss
    loss = acc_loss / n_iterations

    # pixel accuracy
    pix_acc = 1.0 * acc_correct / (np.spacing(1) + acc_labelled)

    # miou
    iou = 1.0 * acc_inter / (np.spacing(1) +  acc_union)

    # save iou list as .csv file
    iou_list = np.array(acc_iou_list)
    df = pd.DataFrame(iou_list, index=[model.name])
    df.to_csv(os.path.join(path, model.name))

            
def test_models(models, dataset, dataloader, path='gen'):
    fcresnet, pspnet = models[0], models[1]

    fcresnet.load_state_dict(torch.load(os.path.join(path, f'{models[0].name}/e_FINAL.pth')))
    pspnet.load_state_dict(torch.load(os.path.join(path, f'{models[1].name}/e_FINAL.pth')))

    fcresnet.eval().to(device)
    pspnet.eval().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    criterion_ignore = nn.CrossEntropyLoss(ignore_index=255).to(device)

    fcrn_acc_loss, pspn_acc_loss = 0, 0
    fcrn_acc_correct, fcrn_acc_labelled, pspn_acc_correct, pspn_acc_labelled = 0, 0, 0, 0
    fcrn_acc_inter, fcrn_acc_union, pspn_acc_inter, pspn_acc_union = 0, 0, 0, 0

    fcrn_iou_acc_list = np.zeros((32, 2), dtype=np.float32)
    pspn_iou_acc_list = np.zeros((32, 2), dtype=np.float32)

    with torch.no_grad():
        n_iterations = int(len(dataset)/dataloader.batch_size)
        progress_bar = tqdm(dataloader, total=n_iterations)

        for i, (data, label) in enumerate(progress_bar):
            data, label = data.to(device), label.to(device)

            out_fcresnet = fcresnet(data)['out']
            fcresnet_loss = criterion(out_fcresnet, label.long())

            out_pspnet = pspnet(data)
            pspnet_loss = criterion_ignore(out_pspnet, label.long())

            fcrn_acc_loss += fcresnet_loss.item()
            pspn_acc_loss += pspnet_loss.item()

            fcrn_C, fcrn_L, fcrn_I, fcrn_U, fcrn_iou_list = eval_metrics(out_fcresnet.data, label.clone().detach(), ret_iou_list=True)
            fcrn_I, fcrn_U = torch.from_numpy(fcrn_I).to(device), torch.from_numpy(fcrn_U).to(device)
            pspn_C, pspn_L, pspn_I, pspn_U, pspn_iou_list = eval_metrics(out_pspnet.data, label.clone().detach(), ret_iou_list=True)
            pspn_I, pspn_U = torch.from_numpy(pspn_I).to(device), torch.from_numpy(pspn_U).to(device)

            # accummulate pixel accuracy
            fcrn_acc_correct += fcrn_C
            fcrn_acc_labelled += fcrn_L

            pspn_acc_correct += pspn_C
            pspn_acc_labelled += pspn_L

            # accummulate IoU
            fcrn_acc_inter += fcrn_I
            fcrn_acc_union += fcrn_U

            pspn_acc_inter += pspn_I
            pspn_acc_union += pspn_U

            # accummulate IoU list
            fcrn_iou_acc_list += fcrn_iou_list
            pspn_iou_acc_list += pspn_iou_list


        # average loss
        fcrn_loss = fcrn_acc_loss / n_iterations
        pspn_loss = pspn_acc_loss / n_iterations

        # pixel accuracy
        fcrn_pix_accu = 1.0 * fcrn_acc_correct / (np.spacing(1) + fcrn_acc_labelled)
        pspn_pix_accu = 1.0 * pspn_acc_correct / (np.spacing(1) + pspn_acc_labelled)

        # mean IoU
        fcrn_iou = 1.0 * fcrn_acc_inter / (np.spacing(1) +  fcrn_acc_union)
        pspn_iou = 1.0 * pspn_acc_inter / (np.spacing(1) + pspn_acc_union)
        fcrn_miou, pspn_miou = fcrn_iou.mean(), pspn_iou.mean()

        iou_list = [fcrn_iou_acc_list, pspn_iou_acc_list]
        ioulist_to_csv(iou_list, path)

        

if __name__ == '__main__':
    fm = fcn_resnet_model
    psp = psp_b1_max

    metrics = load_metrics_from_file('psp_variants/pspnet_b1_max/e_FINAL.pth')

    #test_model(psp, test_dataset, test_dataloader, path='gen')
    #test_models([fm, psp], test_dataset, test_dataloader, path='gen_gamma0.7')