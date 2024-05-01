import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

import config
from camvid import CamVid
from utils.helpers import draw_seg_map
from utils.metrics import eval_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    MODEL_PATH = 'models/'
    CHECKPOINT_PATH = 'models/checkpoints'

    def __init__(self,
                 model, 
                 train_dataset, train_dataloader,
                 val_dataset, val_dataloader,
                 epochs=config.EPOCHS, lr=config.LR,
                 num_classes=len(config.CLASSES_TO_TRAIN)
                 ):
        self.model = model

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader

        self.epochs = epochs
        self.lr = lr

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.num_classes = num_classes

    def fit(self):
        self.model.train()

        n_iterations = int(len(self.train_dataset)/self.train_dataloader.batch_size)
        progress_bar = tqdm(self.train_dataloader, total=n_iterations)

        batch_counter = 0

        train_loss = 0
        train_acc_I, train_acc_U = 0, 0

        for i, (input, label) in enumerate(progress_bar):
            batch_counter += 1

            input = input.to(device)
            label = label.to(device)

            self.optimizer.zero_grad()

            with torch.autograd.set_detect_anomaly(True):
                output = self.model(input)['out']

                # COMPUTE LOSS 
                loss = self.criterion(output, label)

                # EVALUATE METRICS
                # loss
                train_loss += loss.item()

                train_I, train_U = eval_metrics(output.data, torch.tensor(label), self.num_classes)
                train_I = torch.from_numpy(train_I).to(device)
                train_U = torch.from_numpy(train_U).to(device)

                # IoU
                train_acc_I += train_I
                train_acc_U += train_U

                # BACKPROPAGATE AND UPDATE PARAMETERS
                loss.backward()
                self.optimizer.step()

            progress_bar.set_description(f'TRAINING loss: {loss:.3f} | {train_loss/batch_counter:.3f}')

        print()

        # loss
        train_average_loss = train_loss / batch_counter

        # mean intersection over union
        IoU = 1.0 * train_acc_I / (np.spacing(1) + train_acc_U) # spacing to ensure non-zero union
        mIoU = IoU.mean()


        return train_average_loss, mIoU

    def validate(self, epoch):
        self.model.eval()

        val_loss = 0
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

                output = self.model(input)['out']

                # ... draw segmentation map
                # on last batch
                if i == n_iterations - 1:
                    draw_seg_map(input, label, output, epoch)

                # COMPUTE LOSS
                loss = self.criterion(output, label)
            
                # EVALUATE METRICS
                # loss
                val_loss += loss.item()

                val_I, val_U = eval_metrics(output.data, torch.tensor(label), self.num_classes)
                val_I = torch.from_numpy(val_I).to(device)
                val_U = torch.from_numpy(val_U).to(device)

                # IoU
                val_acc_I += val_I
                val_acc_U += val_U

                progress_bar.set_description(desc=f'VALIDATE loss: {loss:.3f} | {val_loss/batch_counter:.3f}')
            
        print()
        
        # loss 
        val_average_loss = val_loss/batch_counter

        # mean intersection over union
        IoU = 1.0 * val_acc_I / (np.spacing(1) + val_acc_U) # spacing to ensure non-zero union
        mIoU = IoU.mean()

        return val_average_loss, mIoU

    def save_checkpoint(self, epoch):
        print(f'SAVING CHECKPOINT ON EPOCH {epoch+1}')
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'loss' : self.criterion
        },  
            os.path.join(self.CHECKPOINT_PATH, f'{CamVid.NAME}_{epoch}.pth')
        )
    
    def save_model(self):
        print(f'SAVING MODEL')
        torch.save(
            self.model.state_dict(), 
            os.path.join(self.MODEL_PATH, f'{CamVid.NAME}_FINAL.pth')
        )