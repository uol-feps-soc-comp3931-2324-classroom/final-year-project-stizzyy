import config

from utils.helpers import make_dirs
from camvid import train_dataset, train_dataloader
from camvid import val_dataset, val_dataloader
from trainer import Trainer, device
from architectures.fcn_resnet50 import Model


def train_and_validate(model):
    trainer = Trainer(model, train_dataset, train_dataloader, val_dataset, val_dataloader)
    
    loss = { 'train' : [], 'val' : [] }
    pix_acc = { 'train' : [], 'val' : [] }
    mIoU = { 'train' : [], 'val' : [] }

    epochs = config.EPOCHS
    for epoch in range(epochs):
        print(f'EPOCH {epoch+1}/{epochs}')
        print('-' * 10)
        train_loss, train_pix_acc, train_mIoU = trainer.fit()
        val_loss, val_pix_acc, val_mIoU = trainer.validate(epoch)

        loss['train'].append(train_loss)
        loss['val'].append(val_loss)

        pix_acc['train'].append(train_pix_acc)
        pix_acc['val'].append(val_pix_acc)
        
        mIoU['train'].append(train_mIoU)
        mIoU['val'].append(val_mIoU)

        if epoch % config.SAVE_CHECKPOINT == 0:
            trainer.save_checkpoint(epoch)
        
        print()

    trainer.save_model()
        


if __name__ == '__main__':
    make_dirs()
    model = Model()
    model = model.to(device)

    train_and_validate(model)