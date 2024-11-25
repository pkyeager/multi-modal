import os
import shutil
import csv

# Define the paths
cwd = os.getcwd().split('scripts')[0]
source_dir = cwd+'data/sentinel-1-flipped-one-class/train'
images_dir = os.path.join(source_dir, 'images')
masks_dir = os.path.join(source_dir, 'masks')
csv_file = os.path.join(source_dir, 'image_names.csv')

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# List to store image names
image_names = []

# Iterate through the files in the source directory
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    if os.path.isfile(file_path):
        if filename.endswith('.jpg'):
            print(f"Moving {filename} to images folder")
            shutil.move(file_path, os.path.join(images_dir, filename))
            image_names.append(filename)
        elif filename.endswith('.png'):
            print(f"Moving {filename} to masks folder")
            shutil.move(file_path, os.path.join(masks_dir, filename))

# Write the image names to a CSV file
print(f"Writing image names to {csv_file}")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name'])
    for name in image_names:
        writer.writerow([name])

print("Script completed successfully")

