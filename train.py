import config

from utils.helpers import make_pth_dirs
from camvid import train_dataset, train_dataloader
from camvid import val_dataset, val_dataloader
from trainer import Trainer, device
from architectures.fcn_resnet50 import Model


def train(model):
    trainer = Trainer(model, train_dataset, train_dataloader, val_dataset, val_dataloader)
    
    epochs = config.EPOCHS
    for epoch in range(2):
        print(f'EPOCH {epoch+1}')
        train_loss = trainer.fit()

        if epoch % 5 == 0:
            trainer.save_checkpoint(epoch)

    trainer.save_model()
        


if __name__ == '__main__':
    make_pth_dirs(Trainer.CHECKPOINT_PATH)
    model = Model()
    model = model.to(device)

    train(model)