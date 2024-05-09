import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def visualize_loss(loss : dict, epochs, model_name, path):
    t_loss = np.array(loss['train'])
    v_loss = np.array(loss['val'])
    x = np.arange(len(loss['train']))

    fig, ax = plt.subplots(1, 1, figsize=(10,6), layout='constrained')
    ax.plot(x, t_loss, label='train', color='tab:red')
    ax.plot(x, v_loss, label='val', color='tab:blue')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_xticks(range(0, epochs+1, 5))
    ax.set_ylim(bottom=0)

    ax.grid(color='gray', linestyle='-', alpha=0.4)

    ax.legend(title='Split', fontsize='medium')

    fig.suptitle(f'{model_name.upper()} Training/Validation Loss')

    fig.savefig(os.path.join(path, 'loss.png'))

def visualize_pixacc(pix_acc : dict, epochs, model_name, path):
    t_pix_acc = np.array(pix_acc['train']) * 100
    v_pix_acc = np.array(pix_acc['val']) * 100
    x = np.arange(len(pix_acc['train']))

    fig, ax = plt.subplots(1, 1, figsize=(10,6), layout='constrained')
    ax.plot(x, t_pix_acc, label='train', color='tab:red')
    ax.plot(x, v_pix_acc, label='val', color='tab:blue')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Pixel Accuracy%')
    ax.set_xticks(range(0, epochs+1, 5))
    ax.set_ylim(top=100)

    ax.grid(color='gray', linestyle='-', alpha=0.4)

    ax.legend(title='Split', fontsize='medium')

    fig.suptitle(f'{model_name.upper()} Training/Validation Pixel Accuracy %')

    fig.savefig(os.path.join(path, 'pix_acc.png'))

def visualize_mIoU(miou : dict, epochs, model_name, path):
    t_miou = np.array(miou['train'])
    v_miou = np.array(miou['val'])
    x = np.arange(len(miou['train']))

    fig, ax = plt.subplots(1, 1, figsize=(10,6), layout='constrained')
    ax.plot(x, t_miou, label='train', color='tab:red')
    ax.plot(x, v_miou, label='val', color='tab:blue')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('mIoU')
    ax.set_xticks(range(0, epochs+1, 5))
    ax.set_ylim(bottom=0)

    ax.grid(color='gray', linestyle='-', alpha=0.4)

    ax.legend(title='Split', fontsize='medium')

    fig.suptitle(f'{model_name.upper()} Training/Validation mIoU')

    fig.savefig(os.path.join(path, 'mIoU.png'))


def create_metrics_viz(metrics_list : list, epochs, trainer):
    model_name = trainer.model.name
    path = trainer.FULL_VIZ_PATH
    visualize_loss(metrics_list[0], epochs, model_name, path)
    visualize_pixacc(metrics_list[1], epochs, model_name, path)
    visualize_mIoU(metrics_list[2], epochs, model_name, path)
    plt.close('all')



def ioulist_to_csv(iou_lists, path):
    ious_list= [[], []]
    for i, iou_list in enumerate(iou_lists): # [fcresnet, pspnet]
        for inter, union in iou_list:
            iou = 1.0 * inter / (np.spacing(1) + union)
            ious_list[i].append(iou)
    
    ious_list = np.reshape(ious_list, (32, -1))
    df = pd.DataFrame(ious_list, columns=['FCN-Resnet50', 'PSPNet'])
    df.to_csv(path)


if __name__ == '__main__':
    a = np.random.randn(32, 2)
    b = np.random.randn(32, 2)

    ioulist_to_csv([a,b])
