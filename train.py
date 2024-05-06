import config
from torchvision.models.segmentation import FCN

from utils.visualization import create_metrics_viz
from camvid import train_dataset, train_dataloader
from camvid import val_dataset, val_dataloader
from trainer import Trainer
from models.fcn_resnet50 import fcn_resnet_model
from models.pspnet import psp_model, PSPNet


def train_and_validate(model):
    trainer = Trainer(model, train_dataset, train_dataloader, val_dataset, val_dataloader)
    
    loss = { 'train' : [], 'val' : [] }
    pix_acc = { 'train' : [], 'val' : [] }
    mIoU = { 'train' : [], 'val' : [] }

    metrics = [loss, pix_acc, mIoU]

    epochs = config.EPOCHS
    for epoch in range(epochs+1):
       
        print(f'EPOCH {epoch+1}/{epochs}')
        print('-' * 10)

        if isinstance(trainer.model, FCN):
            train_loss, train_pix_acc, train_mIoU = trainer.fit(model)
        elif isinstance(trainer.model, PSPNet):
            train_loss, train_aux_loss, train_pix_acc, train_mIoU = trainer.fit(model)

        val_loss, val_pix_acc, val_mIoU = trainer.validate(model, epoch)

        loss['train'].append(train_loss)
        loss['val'].append(val_loss)

        pix_acc['train'].append(train_pix_acc)
        pix_acc['val'].append(val_pix_acc)
        
        mIoU['train'].append(train_mIoU)
        mIoU['val'].append(val_mIoU)

        create_metrics_viz(metrics, epochs, trainer)

        if epoch % config.SAVE_CHECKPOINT == 0:
            trainer.save_checkpoint(epoch)
        
        print()

    trainer.save_model()
        


if __name__ == '__main__':
    train_and_validate(fcn_resnet_model)
    train_and_validate(psp_model)