'''
This script is used to run inference with SAM.
The goal is to segment the tiles of images and save the predictions.
'''

import os
import torch
import numpy as np
import cv2
from transformers import SamModel, SamProcessor
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress tracking
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load pre-trained model and checkpoint
device = "cpu"  # Use CPU
model = SamModel.from_pretrained("facebook/sam-vit-base")
checkpoint_path = "models/image-segmentation/sam-5-epochs.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

def get_bounding_box(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0, 0, 0, 0]  # Default box
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return [x_min, y_min, x_max, y_max]

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def process_tile(tile_path):
    tile_image = Image.open(tile_path).convert("RGB")
    prompt = [0, 0, 256, 256]  # Full image box
    inputs = processor(tile_image, input_boxes=[[prompt]], return_tensors="pt").to(device)
    
    for key in inputs.keys():
        if inputs[key].dtype == torch.float64:
            inputs[key] = inputs[key].to(torch.float32)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    pred_masks = outputs.pred_masks
    pred_mask = torch.sigmoid(pred_masks)
    binary_mask = (pred_mask > 0.5).cpu().numpy()
    
    pred_mask = pred_mask.squeeze().cpu().numpy()
    binary_mask = binary_mask.squeeze()
    
    return tile_image, pred_mask, binary_mask

def save_prediction(prediction, output_path):
    print(f"Prediction shape before saving: {prediction.shape}")
    if prediction.ndim > 2:
        prediction = prediction.squeeze()
    prediction_image = Image.fromarray((prediction * 255).astype(np.uint8))
    prediction_image.save(output_path)

def plot_predictions(tile_image, raw_mask, binary_mask):
    plt.figure(figsize=(15, 5))
    print(f"Raw mask shape: {raw_mask.shape}")
    print(f"Binary mask shape: {binary_mask.shape}")
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(np.array(tile_image))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Raw Predicted Mask")
    plt.imshow(raw_mask, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def run_inference_on_tiles(tiles_dir, predictions_base_dir):
    for root, dirs, files in os.walk(tiles_dir):
        for file in tqdm(files, desc="Processing tiles"):
            if file.endswith((".jpg", ".png")):
                tile_path = os.path.join(root, file)
                
                # Process the tile
                tile_image, raw_mask, binary_mask = process_tile(tile_path)
                
                # Plot predictions for the first image only
                if file == files[0]:  # Check if it's the first file
                    plot_predictions(tile_image, raw_mask, binary_mask)
                
                # Create predictions directory
                date_str = os.path.basename(root)
                predictions_dir = os.path.join(predictions_base_dir, date_str, "predictions")
                os.makedirs(predictions_dir, exist_ok=True)
                
                # Save the prediction
                prediction_name = f"pred_{file}"
                prediction_path = os.path.join(predictions_dir, prediction_name)
                save_prediction(binary_mask, prediction_path)

if __name__ == "__main__":
    tiles_dir = "data/sonderborg/2023/tiled"  # Directory containing the tiles
    predictions_base_dir = "data/sonderborg/2023/tiled"  # Base directory for predictions
    run_inference_on_tiles(tiles_dir, predictions_base_dir)


