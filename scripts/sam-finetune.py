import warnings
warnings.filterwarnings("ignore")

from segment_anything import sam_model_registry
from segment_anything import *

import torchvision.transforms as T
import torch
import torch.nn as nn


import torch.distributed as dist
from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


import os
import argparse
import wandb

from datatime import datetime
from torchinfo import summary

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

### Constants ###
IMAGE_SIZE = (1024, 1024)
NUM_CLASSES = 1

# TODO: Can be cahnged later
IMAGE_DIR = "data/sentinel-1-flipped-one-class/train"




### Helper functions ###


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    '''
    TODO: Might have to add more, but for now, this is enough
    '''
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--batch-size', type=int, default=4, help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--dist', type=str2bool, default=False, help='distributed training (default: False)')
    parser.add_argument('--model_type', type=str, default='vit_t', help='SAM model type (default: vit_t)')
    parser.add_argument('--model_checkpoint', type=str, default='sam_vit_t', help='SAM model checkpoint (feauture extractor) (default: sam_vit_t)')

    return parser

### Fine-tuning script ###
def main(opts) -> str:
    """
    Model finetuning script

    Returns:
        str: Path to the saved models
    """

    seed.set_seed(opts.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ### Checkpointing ###

    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = run_time + f'_{opts.model_type}_finetune.pth'
    save_path = os.path.join(CHECKPOINT_DIR, file_name)


    ### Dataset & Dataloader ###

    train_transform = T.Compose([
        T.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        T.ToImage()
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = T.Compose([
        T.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        T.ToImage()
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = dataset.make_dataset(IMAGE_DIR,
                                     transform=train_transform)

    val_set = dataset.make_dataset(IMAGE_DIR,
                                      transform=test_transform)


    train_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=opts.batch_size, shuffle=False)

    ### SAM Config ###
    sam_checkpoint = opts.model_checkpoint
    model_type = opts.model_type

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    ### Freezing the model ###

    for _, p in sam.image_encoder.named_parameters():
        p.requires_grad = False

    for _, p in sam.prompt_encoder.named_parameters():
        p.requires_grad = False

    for _, p in sam.mask_decoder.named_parameters():
        p.requires_grad = True

    # print model info

    print()
    print(" === MODEL INFO ===")
    summary(sam, input_size=(opts.batch_size, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    print()

    model = sam

    ### Loss & Optimizer ###
    bceloss = nn.BCEWithLogitsLoss()
    iouloss = iou_loss_torch.IoULoss().to(device)