from torch.utils.data import Dataset, random_split
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os
import torch
import json

default_data_path = os.path.join(
    os.path.dirname(__file__), '../../../preprocess/Detection_NIH_1024_version_2.csv')


class NIHDataset(Dataset):
    def __init__(self, data_file=default_data_path, normalize_scale=1/1024.0):
        super(NIHDataset, self).__init__()
        self.data = pd.read_csv(data_file)
        self.normalize_scale = normalize_scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        bbox = torch.tensor(json.loads(data['bbox'])) * self.normalize_scale

        label = torch.tensor(json.loads(data['Label']))
        image = transforms.ToTensor()(Image.open(data['image_path']))

        return image, {
            'bboxes': torch.tensor(bbox, dtype=torch.float),
            'class_labels': torch.tensor(label, dtype=torch.long),
        }


def get_dataset(data_file=default_data_path,
                normalize_scale=1/1024.0,
                splits=[0.8, 0.1, 0.1]
                ):
    dataset = NIHDataset(data_file, normalize_scale)
    return random_split(dataset, splits)
