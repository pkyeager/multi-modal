''' 
This utility script should  take all images
in a directory and tile them into a single image. 
All images should be of the same size. 
The output image should be a square image and 
the images should be organized so that we can compare the same tile 
across different images. 
'''

import os
import numpy as np
import cv2
import re

TILE_H = 256
TILE_W = 256

img_dir = "data/sonderborg/2023/"
output_base_dir = "data/sonderborg/2023/tiled"

if not os.path.exists(img_dir):
    print(f"Input directory does not exist: {img_dir}")
    exit(1)  

def get_images(img_dir):
    return [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".tiff")]

def extract_date_from_filename(filename):
    '''
    Extract the date from the filename in the format YYYY-MM-DD.
    '''
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return match.group(1) if match else None

def tile_images(image_paths):
    '''
    Tile all images in the given directory and save the tiles in subdirectories named after the date in the filename.
    '''
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)  # Convert 16-bit to 8-bit
        elif img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)  # Convert float to 8-bit

        img_h, img_w = img.shape[:2]
        
        tile_rows = img_h // TILE_H
        tile_cols = img_w // TILE_W
        
        # Extract the date from the filename
        date_str = extract_date_from_filename(os.path.basename(image_path))
        if date_str is None:
            print(f"Could not extract date from filename: {image_path}")
            continue
        
        # Create a directory for the date if it doesn't exist
        date_output_dir = os.path.join(output_base_dir, date_str)
        if not os.path.exists(date_output_dir):
            os.makedirs(date_output_dir)

        for row in range(tile_rows):
            for col in range(tile_cols):
                tile = img[row*TILE_H:(row+1)*TILE_H, col*TILE_W:(col+1)*TILE_W]
                # Change tile name to just the coordinates
                tile_name = f"tile_{row}_{col}.png"
                tile_path = os.path.join(date_output_dir, tile_name)
                cv2.imwrite(tile_path, tile)

if __name__ == "__main__":
    image_paths = get_images(img_dir)
    tile_images(image_paths)
