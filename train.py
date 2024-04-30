import config

from utils.helpers import make_pth_dirs
from camvid import train_dataset, train_dataloader
from camvid import val_dataset, val_dataloader
from trainer import Trainer, device
from architectures.fcn_resnet50 import Model


def train_and_validate(model):
    trainer = Trainer(model, train_dataset, train_dataloader, val_dataset, val_dataloader)
    
    epochs = config.EPOCHS
    for epoch in range(epochs):
        print(f'EPOCH {epoch+1}/{epochs}')
        print('-' * 10)
        train_loss, train_mIoU = trainer.fit()
        val_loss, val_mIoU = trainer.validate(epoch)

        if epoch % config.SAVE_CHECKPOINT == 0:
            trainer.save_checkpoint(epoch)
        
        print()

    trainer.save_model()
        


if __name__ == '__main__':
    make_pth_dirs(Trainer.CHECKPOINT_PATH)
    model = Model()
    model = model.to(device)

    train_and_validate(model)