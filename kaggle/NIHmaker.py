import os
import pandas as pd
from PIL import Image
import shutil
import json
from sklearn.model_selection import train_test_split

def df_to_coco(df, output_file, categories=None):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    if categories is None:
        categories = list(set(df['Finding Label'].tolist()))

    category_ids = {cat: i+1 for i, cat in enumerate(categories)}

    # Fill categories
    for category in categories:
        coco_format['categories'].append({
            "id": category_ids[category],
            "name": category,
            "supercategory": "none",
        })

    annotation_id = 1

    for index, row in df.iterrows():
        image_id = index + 1  # Assuming each row is a unique image
        image_info = {
            "id": image_id,
            "file_name": os.path.basename(row['image_path']),
            "height": 1024,  # Replace with actual height
            "width": 1024,   # Replace with actual width
        }

        coco_format['images'].append(image_info)

        bbox = [row['Bbox [x'], row['y'], row['w'], row['h]']]
        bbox = [float(x) for x in bbox]  # Convert to float, if necessary
        area = bbox[2] * bbox[3]

        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_ids[row['Finding Label']],
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        }

        coco_format['annotations'].append(annotation_info)
        annotation_id += 1

    # Save to JSON file
    with open(output_file, 'w') as outfile:
        json.dump(coco_format, outfile, indent=4)

    print(f"COCO format JSON saved to {output_file}")

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        mode = img.mode
        channels = len(mode)
    return f"({channels},{width},{height})"

def get_image_dict():
    all_images = {}
    image_number = 0
    for i in range(1, 13):
        current_folder_image = {}
        folder_path = f'/kaggle/input/data/images_{str(i).zfill(3)}/images'
        image_files = os.listdir(folder_path)

        current_folder_image = {k: os.path.join(
            folder_path, k) for k in image_files}

        print(f'Folder {i}:', len(current_folder_image), len(
            image_files), f'Valid: {len(current_folder_image) == len(image_files)}')
        image_number += len(current_folder_image)

        all_images.update(current_folder_image)

    print('All image:', len(all_images), image_number,
          f'Valid: {len(all_images) == image_number}')

    return all_images

def get_detection_data(all_image, train_dir='./train', val_dir='./val', split=0.9):
    bbox_list = pd.read_csv('/kaggle/input/data/BBox_List_2017.csv')
    bbox_list = bbox_list[['Image Index',
                           'Finding Label', 'Bbox [x', 'y', 'w', 'h]']]
    bbox_list['image_path'] = bbox_list['Image Index'].map(all_image)    
    train_df, test_df = train_test_split(bbox_list, test_size=0.1, random_state=42)
    
    print(len(test_df))
    
    os.makedirs(train_dir)
    for src_file in train_df['image_path']:
        dest_file = os.path.join(train_dir, os.path.basename(src_file))
        shutil.copy2(src_file, dest_file)
        
    os.makedirs(val_dir)
    for src_file in test_df['image_path']:
        dest_file = os.path.join(val_dir, os.path.basename(src_file))
        shutil.copy2(src_file, dest_file)
        
    os.makedirs('./annotations')
    df_to_coco(train_df, './annotations/train.json')
    df_to_coco(test_df, './annotations/val.json')
    
    return bbox_list


data = get_detection_data(get_image_dict())

