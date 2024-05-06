import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

import config
from utils.helpers import draw_seg_map
from utils.metrics import eval_metrics
from torchvision.models.segmentation import FCN
from models.pspnet import PSPNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    GEN_PATH = 'gen'
    CHECKPOINT_PATH = 'checkpoints'
    SEGMAP_PATH = 'segmaps'
    VIZ_PATH = 'viz'

    def __init__(self, model, 
                 train_dataset, train_dataloader,
                 val_dataset, val_dataloader,
                 epochs=config.EPOCHS, lr=config.LR,
                 num_classes=len(config.CLASSES_TO_TRAIN)
                 ):
        self.model = model # do not use self.model for fit and validate

        self.FULL_GEN_PATH = os.path.join(self.GEN_PATH, self.model.name)
        self.FULL_CHECKPOINT_PATH = os.path.join(self.GEN_PATH, self.model.name, self.CHECKPOINT_PATH)
        self.FULL_SEGMAP_PATH = os.path.join(self.GEN_PATH, self.model.name, self.SEGMAP_PATH)
        self.FULL_VIZ_PATH = os.path.join(self.GEN_PATH, self.model.name, self.VIZ_PATH)

        for path in [self.FULL_GEN_PATH, self.FULL_CHECKPOINT_PATH, self.FULL_SEGMAP_PATH, self.FULL_VIZ_PATH]:
            os.makedirs(path, exist_ok=True)

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader

        self.epochs = epochs
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.num_classes = num_classes

    def fit(self, model):
        model.train()

        n_iterations = len(self.train_dataset)//self.train_dataloader.batch_size
        progress_bar = tqdm(self.train_dataloader, total=n_iterations)

        batch_counter = 0

        train_loss, train_aux_loss, aux_loss = 0, 0, -1
        train_acc_correct, train_acc_label = 0, 0
        train_acc_I, train_acc_U = 0, 0

        for i, (input, label) in enumerate(progress_bar):
            batch_counter += 1

            input = input.to(device)
            label = label.to(device)

            self.optimizer.zero_grad()

            with torch.autograd.set_detect_anomaly(True):
                if isinstance(model, FCN):
                    output = model(input)['out']

                    # COMPUTE LOSS 
                    loss = self.criterion(output, label)
                
                elif isinstance(model, PSPNet):
                    output, loss, aux_loss = model(input, label.long())

                    # COMPUTE LOSS
                    train_aux_loss += aux_loss.item()

                train_loss += loss.item()

                # EVALUATE METRICS
                train_C, train_L, train_I, train_U = eval_metrics(output.data, label.clone().detach(), self.num_classes)
                train_I = torch.from_numpy(train_I).to(device)
                train_U = torch.from_numpy(train_U).to(device)

                # Pixel Accuracy
                train_acc_correct += train_C
                train_acc_label += train_L

                # IoU
                train_acc_I += train_I
                train_acc_U += train_U

                # BACKPROPAGATE AND UPDATE PARAMETERS
                loss.backward()
                self.optimizer.step()

                # LOGGING
                train_curr_pixaccuracy = 1.0 * train_C / (np.spacing(1) + train_L)

                train_curr_iou = 1.0 * train_I / (np.spacing(1) + train_U)
                train_curr_miou = train_curr_iou.mean()

                progress_bar.set_description(f'TRAINING-- loss: {loss:.3f} | aux loss: {aux_loss:.3f} | pix acc: {train_curr_pixaccuracy:.3f} | mIoU: {train_curr_miou:.3f}')

        print()

        # loss
        t_loss = train_loss / batch_counter
        t_aux_loss = train_aux_loss / batch_counter

        # pixel accuracy
        pix_acc = 1.0 * train_acc_correct / (np.spacing(1) + train_acc_label) # spacing to ensure non-zero union

        # mean intersection over union
        IoU = 1.0 * train_acc_I / (np.spacing(1) + train_acc_U) # spacing to ensure non-zero union
        mIoU = IoU.mean().cpu().numpy().min()

        if isinstance(model, PSPNet):
            return t_loss, t_aux_loss, pix_acc, mIoU
        else:
            return t_loss, pix_acc, mIoU

    def validate(self, model, epoch):
        model.eval()

        val_loss = 0
        val_acc_correct, val_acc_label = 0, 0
        val_acc_I, val_acc_U = 0, 0

        batch_counter = 0

        n_iterations = int(len(self.val_dataset)/self.val_dataloader.batch_size)

        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, total=n_iterations)

            for i, (input, label) in enumerate(progress_bar):
                batch_counter += 1

                input = input.to(device)
                label = label.to(device)

                self.optimizer.zero_grad()

                output = model(input)
                if isinstance(model, FCN):
                    output = output['out']

                # ... draw segmentation map
                # on last batch
                if i == n_iterations - 1:
                    draw_seg_map(input, label, output, epoch, path=self.FULL_SEGMAP_PATH)

                # COMPUTE LOSS
                loss = self.criterion(output, label)
            
                # EVALUATE METRICS
                # loss
                val_loss += loss.item()

                val_C, val_L, val_I, val_U = eval_metrics(output.data, label.clone().detach(), self.num_classes)
                val_I = torch.from_numpy(val_I).to(device)
                val_U = torch.from_numpy(val_U).to(device)

                # pixel accuracy
                val_acc_correct += val_C
                val_acc_label += val_L

                # IoU
                val_acc_I += val_I
                val_acc_U += val_U

                # LOGGING
                val_curr_pixaccuracy = 1.0 * val_C / (np.spacing(1) + val_L)

                val_curr_iou = 1.0 * val_I / (np.spacing(1) + val_U)
                val_curr_miou = val_curr_iou.mean()

                progress_bar.set_description(f'VALIDATION-- loss: {loss:.3f} | pix acc: {val_curr_pixaccuracy:.3f} | mIoU: {val_curr_miou:.3f}')
            
        print()
        
        # loss 
        v_loss = val_loss/batch_counter

        # pixel accuracy
        pix_acc = 1.0 * val_acc_correct / (np.spacing(1) + val_acc_label) # spacing to ensure non-zero union
        # mean intersection over union
        IoU = 1.0 * val_acc_I / (np.spacing(1) + val_acc_U) # spacing to ensure non-zero union
        mIoU = IoU.mean().cpu().numpy().min()

        return v_loss, pix_acc, mIoU

    def save_checkpoint(self, epoch):
        print(f'SAVING CHECKPOINT ON EPOCH {epoch+1}')
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'loss' : self.criterion
        },  
            os.path.join(self.FULL_CHECKPOINT_PATH, f'e_{epoch}.pth')
        )
    
    def save_model(self):
        print(f'SAVING MODEL')
        torch.save(
            self.model.state_dict(), 
            os.path.join(self.FULL_GEN_PATH, f'e_FINAL.pth')
        )