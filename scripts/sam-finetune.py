'''
Use MedSAM as inspiration to fine-tune our model for the Sentinel-1 dataset.
'''

### Setup the environment ###
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
join = os.path.join
from tdqm import tqdm
from skimage.io import transform
import torch
import torch.nn as nn


### Setup Parser ###
# TODO: For later use, could be quite nice to have a parser for the script
# parser = argparse.ArgumentParser()


### Constants ###
SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data/sentinel-1-flipped-one-class/train"
IMAGES_DIR = join(DATA_DIR, "images")
MASKS_DIR = join(DATA_DIR, "masks")
TEST_SIZE = 0.2

### Utility Functions ###
def show_mask(mask, ax, random_color=False):
    color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

### Custom Dataset Class ###
class SARDataset(Dataset):
    def __init__(self, data_root, bbox_shift = 20):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

train_dataset = SARDataset(DATA_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

### Sanity Check ###
for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
    print(image.shape, gt.shape, bboxes.shape)
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, 7)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    # set title
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 7)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")
    # set title
    axs[1].set_title(names_temp[idx])
    # plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("../data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    break

class SARSam(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):

def main():


if __name__ == "__main__":
    main()