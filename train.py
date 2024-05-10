import config
import torch
from torch import optim
from torchvision.models.segmentation import FCN

from utils.visualization import create_metrics_viz
from camvid import train_dataset, train_dataloader
from camvid import val_dataset, val_dataloader
from trainer import Trainer, device
from models.pspnet import base, psp_notpretrained, psp_b1_avg, psp_b1_max, psp_b1236_avg, psp_b1236_max
from models.pspnet import PSPNet


optimizers = {
    'adam' : [ optim.Adam, {'weight_decay' : 0.0001 }],
    'sgd' : [ optim.SGD, {'weight_decay' : 0.0001, 'momentum' : 0.9 }]
}

schedulers = {
    'exponential_g07' : [ optim.lr_scheduler.ExponentialLR, { 'gamma' : 0.7 }],
    'exponential_g095' : [ optim.lr_scheduler.ExponentialLR, { 'gamma' : 0.95 }],
    'poly' : [optim.lr_scheduler.PolynomialLR, { 'total_iters' : config.EPOCHS, 'power' : 0.09}]
}


def train_and_validate(model, opt='adam', sch='exponential_g095'):
    opt_cls, sch_cls = optimizers[opt][0], schedulers[sch][0] # classes
    opt_p, sch_p = optimizers[opt][1], schedulers[sch][1] # parameters

    optimizer = opt_cls(model.parameters(), lr=config.LR, **opt_p)
    scheduler = sch_cls(optimizer, **sch_p)

    trainer = Trainer(model, train_dataset, train_dataloader, val_dataset, val_dataloader, optimizer=optimizer)

    # warm up device
    dummy_data = torch.randn(1, 3, 224, 224).to(device)
    print(f'WARMING UP {str(device).upper()}')
    model.eval()
    for i in range(11):
        _ = model(dummy_data)

    loss = { 'train' : [], 'val' : [] }
    pix_acc = { 'train' : [], 'val' : [] }
    mIoU = { 'train' : [], 'val' : [] }
    metrics = [loss, pix_acc, mIoU]

    print(model.name)
    print('-' * 10)
    epochs = config.EPOCHS
    for epoch in range(epochs+1):
       
        print(f'EPOCH {epoch+1}/{epochs+1}')
        print('-' * 10)

        if isinstance(trainer.model, FCN):
            train_loss, train_pix_acc, train_mIoU = trainer.fit(model)
        elif isinstance(trainer.model, PSPNet):
            train_loss, train_aux_loss, train_pix_acc, train_mIoU = trainer.fit(model)

        batches = [0, (config.BATCH_SIZE // 2) - 1, config.BATCH_SIZE - 1]      # [0, 7, 15]
        val_loss, val_pix_acc, val_mIoU = trainer.validate(model, epoch, batches=batches)

        scheduler.step()
        print(scheduler.get_last_lr())

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

    trainer.save_model(metrics)
        


if __name__ == '__main__':
    # training and validating PSPNet variants
    train_and_validate(psp_b1236_max)
