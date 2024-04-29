import torch
import torch.nn as nn
import os
import config
from tqdm import tqdm

from camvid import CamVid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    MODEL_PATH = 'models/'
    CHECKPOINT_PATH = 'models/checkpoints'

    def __init__(self,
                 model, 
                 train_dataset, train_dataloader,
                 val_dataset, val_dataloader,
                 epochs=config.EPOCHS, lr=config.LR, batchsize = config.BATCH_SIZE
                 ):
        self.model = model

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader

        self.epochs = epochs
        self.lr = lr
        self.batchsize = batchsize

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss().cuda()

    def fit(self):
        self.model.train()

        n_iterations = int(len(self.train_dataset)/self.batchsize)
        progress_bar = tqdm(self.train_dataloader, total=n_iterations)

        batch_counter = 0

        train_total_loss = 0

        for i, (input, label) in enumerate(progress_bar):
            batch_counter += 1

            input = input.to(device)
            label = label.to(device)

            self.optimizer.zero_grad()

            outputs = self.model(input)['out']

            loss = self.criterion(outputs, label)
            train_total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            progress_bar.set_description(f'loss {loss:.3f} > {train_total_loss/batch_counter:.3f}')

        train_average_loss = train_total_loss / batch_counter

        return train_average_loss

    def validate(self, epoch):
        pass

    def save_checkpoint(self, epoch):

        """ if not os.path.exists(self.CHECKPOINT_PATH):
            os.makedirs(self.CHECKPOINT_PATH)"""

        torch.save({
            'epoch' : epoch,
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'loss' : self.criterion
        },  
            os.path.join(self.CHECKPOINT_PATH, f'e_{epoch}.pth')
        )
    
    def save_model(self):
        """if not os.path.exists(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)"""

        torch.save(
            self.model.state_dict(), 
            os.path.join(self.MODEL_PATH, f'{CamVid.NAME}.pth')
        )