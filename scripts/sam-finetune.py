'''
Used MedSam as inspiration to fine-tune our model for the Sentinel-1 dataset.
'''

### Setup the environment ###
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from tqdm import tqdm



### Setup Parser ###
# TODO: For later use, could be quite nice to have a parser for the script
# parser = argparse.ArgumentParser()



### Constants ###
SEED = 42
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/sentinel-1-flipped-one-class/train"
TEST_SIZE = 0.2
MODEL_NAME = "sam-tiny"
CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
EPOCHS = 20
NUM_WORKERS = 4
RESUME = "checkpoints/resume"
TASK_NAME = "sam-finetune"

## TODO: Replace this with Argparse later
args = argparse.Namespace()
args.use_amp = False
args.use_wandb = False


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
        self.data_root = data_root
        self.mask_path = join(data_root, "masks")
        self.imag_path = join(data_root, "images")
        self.mask_path_files = sorted(glob.glob(join(self.mask_path, "*.png")))
        self.imag_path_files = sorted(glob.glob(join(self.imag_path, "*.jpg")))
        self.bbox_shift = bbox_shift
        print(f"Found {len(self.mask_path_files)} masks and {len(self.imag_path_files)} images")

    def __len__(self):
        return len(self.mask_path_files)

    def __getitem__(self, idx):
        img_name = os.path.basename(self.imag_path_files[idx])
        img_1024 = transform.resize(
            plt.imread(self.imag_path_files[idx]), (1024, 1024)
        )
        img_np = img_1024.transpose(2, 0, 1)
        assert(np.max(img_np) <= 1.0 and np.min(img_np) >= 0.0), f"Image {img_name} has values outside [0, 1]"
        mask_1024 = transform.resize(
            plt.imread(self.mask_path_files[idx]), (1024, 1024)
        )
        mask_np = mask_1024[None, :, :]
        assert(np.max(mask_np) <= 1.0 and np.min(mask_np) >= 0.0), f"Mask {img_name} has values outside [0, 1]"

        ## The box is just the bounding box of the mask
        box = torch.tensor(np.array([0, 0, 1024, 1024]))

        return (torch.tensor(img_np, dtype=torch.float32), torch.tensor(mask_np, dtype=torch.float32), box, img_name)

train_dataset = SARDataset(DATA_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

### Sanity Check ###
for step, (image, gt, bboxes, names_temp) in enumerate(train_loader):
    print(image.shape, gt.shape, bboxes.shape)
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, BATCH_SIZE - 1)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    # set title
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, BATCH_SIZE - 1)
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

### Setup the Model for the training ###
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_dir = join("../checkpoints", run_id + MODEL_NAME)
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = join(model_save_dir, "model.pth")

class SARSam(nn.Module):
    def __init__(self,
                 image_encoder,
                 mask_decoder,
                 prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # Freeze the prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False


    def forward(self, image, box):
        # Add debug prints
        image_embedding = self.image_encoder(image) 
        print("image_embedding:", image_embedding.shape)
        
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            print("Before repeat - dense:", dense_embeddings.shape)
            print("Before repeat - sparse:", sparse_embeddings.shape)
        
        # Add shape checks
        if dense_embeddings.size(0) != image_embedding.size(0):
            print(f"Mismatch: dense={dense_embeddings.size(0)} vs image={image_embedding.size(0)}")
            dense_embeddings = dense_embeddings.repeat(image_embedding.size(0), 1, 1, 1)
            sparse_embeddings = sparse_embeddings.repeat(image_embedding.size(0), 1, 1)

        # Move the batch size check before the mask decoder call
        if dense_embeddings.size(0) != image_embedding.size(0):
            dense_embeddings = dense_embeddings.repeat(image_embedding.size(0), 1, 1, 1)
            sparse_embeddings = sparse_embeddings.repeat(image_embedding.size(0), 1, 1)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():
    os.makedirs(model_save_dir, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_dir, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
    sarsam_model = SARSam(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(DEVICE)
    sarsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in sarsam_model.parameters()),
    )
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in sarsam_model.parameters() if p.requires_grad),
    )

    img_mask_encdec_params = list(sarsam_model.image_encoder.parameters()) + list(
        sarsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )
    seg_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    ### Training Loop ###
    num_epochs = EPOCHS
    iter_num = 0
    losses = []
    best_loss = 1e10
    train_dataset = SARDataset(DATA_DIR)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    start_epoch = 0
    if RESUME is not None:
        if os.path.isfile(RESUME):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(RESUME, map_location=DEVICE)
            start_epoch = checkpoint["epoch"] + 1
            sarsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, names_temp) in enumerate(tqdm(train_dataloader)):
            print(f"image: {image.shape}, gt2D: {gt2D.shape}, boxes: {boxes.shape}")
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(DEVICE), gt2D.to(DEVICE)
            if args.use_amp:
                ## TODO: Add AMP support, what is AMP?
                ## AMP
                with torch.autocast(DEVICE_type="cuda", dtype=torch.float16):
                    sarsam_pred = sarsam_model(image, boxes_np)
                    loss = seg_loss(sarsam_pred, gt2D) + ce_loss(
                        sarsam_pred, gt2D.float()
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                sarsam_pred = sarsam_model(image, boxes_np)
                loss = seg_loss(sarsam_pred, gt2D) + ce_loss(sarsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)
        ## TODO: Add Wandb logging
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        checkpoint = {
            "model": sarsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_dir, "sarsam_model_latest.pth"))
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": sarsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_dir, "sarsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_dir, TASK_NAME + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()