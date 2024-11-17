'''
This script performs image segmentation using the Segment Anything Model (SAM).
It processes image tiles, generates segmentation masks, and saves the predictions.

Key functionalities:
- Load a pre-trained SAM model and its checkpoint.
- Process image tiles to generate raw and binary masks.
- Save the generated masks to specified directories.
- Plot the original images alongside their predicted masks for visual validation.

Usage:
1. Set the `tiles_dir` to the directory containing the image tiles.
2. Set the `predictions_base_dir` to the base directory where predictions will be saved.
3. Run the script to process the tiles and save the predictions.

'''

import os
import torch
import numpy as np
import cv2
from transformers import SamModel, SamProcessor
from PIL import Image
from tqdm import tqdm  

# Load pre-trained model and checkpoint
device = "cpu"  # Use CPU for inference, tried mps but it was buggy with float64
model = SamModel.from_pretrained("facebook/sam-vit-base")
checkpoint_path = "models/image-segmentation/sam-5-epochs.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

def get_bounding_box(ground_truth_map):
    """
    Calculate the bounding box coordinates from the ground truth mask.
    TODO: Is this really necessary?
    Parameters:
    ground_truth_map (numpy.ndarray): The ground truth mask.

    Returns:
    list: A list containing the coordinates of the bounding box [x_min, y_min, x_max, y_max].
    """
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0, 0, 0, 0]  # Default box if no indices found
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return [x_min, y_min, x_max, y_max]

# Initialize the processor for SAM
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def process_tile(tile_path):
    """
    Process a single image tile to generate segmentation masks.

    Parameters:
    tile_path (str): The file path of the image tile.

    Returns:
    tuple: A tuple containing the binary mask only (removed original tile and raw mask from return).
    """
    tile_image = Image.open(tile_path).convert("RGB")
    prompt = [0, 0, 256, 256]  # Define the prompt as the full image box
    inputs = processor(tile_image, input_boxes=[[prompt]], return_tensors="pt").to(device)
    
    # Ensure all input tensors are of type float32
    for key in inputs.keys():
        if inputs[key].dtype == torch.float64:
            inputs[key] = inputs[key].to(torch.float32)
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    pred_masks = outputs.pred_masks
    pred_mask = torch.sigmoid(pred_masks)  # Apply sigmoid to get probabilities
    binary_mask = (pred_mask > 0.5).cpu().numpy()  # Convert to binary mask
    binary_mask = binary_mask.squeeze()  # Remove singleton dimensions
    
    return binary_mask

def save_prediction(prediction, output_path):
    """
    Save the predicted mask as an image file.

    Parameters:
    prediction (numpy.ndarray): The predicted mask to save.
    output_path (str): The file path where the prediction will be saved.
    """
    print(f"Prediction shape before saving: {prediction.shape}")
    if prediction.ndim > 2:
        prediction = prediction.squeeze()  # Squeeze if more than 2 dimensions
    prediction_image = Image.fromarray((prediction * 255).astype(np.uint8))  # Convert to uint8
    prediction_image.save(output_path)  # Save the image

def run_inference_on_tiles(tiles_dir, predictions_base_dir):
    """
    Run inference on all image tiles in the specified directory.

    Parameters:
    tiles_dir (str): The directory containing the image tiles.
    predictions_base_dir (str): The base directory for saving predictions.
    """
    for root, dirs, files in os.walk(tiles_dir):
        for file in tqdm(files, desc="Processing tiles"):
            if file.endswith((".jpg", ".png")):  # Process only image files
                tile_path = os.path.join(root, file)
                
                # Process the tile to get predictions
                binary_mask = process_tile(tile_path)
                
                # Create predictions directory based on the date
                date_str = os.path.basename(root)
                predictions_dir = os.path.join(predictions_base_dir, date_str, "predictions")
                os.makedirs(predictions_dir, exist_ok=True)
                
                # Save the prediction
                prediction_name = f"pred_{file}"
                prediction_path = os.path.join(predictions_dir, prediction_name)
                save_prediction(binary_mask, prediction_path)

if __name__ == "__main__":
    # Define the directories for input tiles and output predictions
    tiles_dir = "data/sonderborg/2023/tiled"  # Directory containing the tiles
    predictions_base_dir = "data/sonderborg/2023/tiled"  # Base directory for predictions
    run_inference_on_tiles(tiles_dir, predictions_base_dir)  # Start the inference process

