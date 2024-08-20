import pandas as pd
import json
import os
from tqdm import tqdm
from ..MedicalVision.utils.spliter import split_train_test

# Load the CSV file
csv_file = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv'  # Replace with your CSV file path
dicom_dir = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train'  # Replace with your DICOM images directory
output_json = 'coco_dataset.json'  # Path to save the COCO format JSON file

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Initialize the COCO format structure
coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Create a dictionary to store category IDs and names
category_map = {}

image_id = 1
annotation_id = 1

# Iterate over the rows in the CSV
for _, row in tqdm(df.iterrows()):
    image_name = row['image_id'] + '.dicom'
    image_path = os.path.join(dicom_dir, image_name)
    
    x_min, y_min = row['x_min'], row['y_min']
    x_max, y_max = row['x_max'], row['y_max']
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    # Add image information if not already added
    if image_id not in [img['id'] for img in coco["images"]]:
        image_info = {
            "id": image_id,
            "file_name": image_name,
            "width": width,   # Placeholder, update if available
            "height": height   # Placeholder, update if available
        }
        coco["images"].append(image_info)

    # Add category if not already added
    class_id = row['class_id']
    class_name = row['class_name']
    if class_id not in category_map:
        category_info = {
            "id": class_id+1,
            "name": class_name,
            "supercategory": "none"
        }
        coco["categories"].append(category_info)
        category_map[class_id] = class_name

    # Add annotation information    
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": class_id,
        "bbox": [x_min, y_min, width, height],
        "area": area,
        "iscrowd": 0
    }
    coco["annotations"].append(annotation)

    annotation_id += 1
    image_id += 1

# Save the COCO JSON structure to a file
with open(output_json, 'w') as f:
    json.dump(coco, f)

split_train_test(output_json, 'train.json', 'test.json', split=0.8)
split_train_test('./test.json', 'mini_train.json', 'mini_val.json', split=0.8)
