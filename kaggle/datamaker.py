import os
import pandas as pd
from PIL import Image


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


def get_detection_data(all_image):
    bbox_list = pd.read_csv('/kaggle/input/data/BBox_List_2017.csv')
    bbox_list = bbox_list[['Image Index',
                           'Finding Label', 'Bbox [x', 'y', 'w', 'h]']]
    bbox_list['image_path'] = bbox_list['Image Index'].map(all_image)
    return bbox_list


data = get_detection_data(get_image_dict())

data.to_csv('./Detection_NIH_1024.csv')
