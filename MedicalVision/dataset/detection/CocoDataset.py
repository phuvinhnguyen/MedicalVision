import torchvision
from torch.utils.data import DataLoader
from ...utils import split_train_test
from PIL import Image
import pydicom
import numpy as np
import os

def get_collator(processor):
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        if "pixel_mask" in encoding:
            batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch
    return collate_fn

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self,
                 img_folder,
                 ann_file,
                 processor):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        if path.endswith(".dicom"):
            image_array = pydicom.dcmread(path).pixel_array
            return Image.fromarray(((image_array / np.max(image_array)) * 255).astype(np.uint8)).convert("RGB")
        else:
            return Image.open(os.path.join(self.root, path)).convert("RGB")
    
def get_loader(
    img_folder,
    processor,
    annotations_file,
    batch_size=16,
    num_workers=0,
    splits=None,
    shuffle=True,
    ):
    datasets = []
    if splits is not None:
        for i, split in enumerate(splits):
            split_train_test(annotations_file, f'./{i}.json', f'./{i+1}.json', split=split)
            datasets.append(CocoDetection(img_folder, f'./{i}.json', processor))
            annotations_file = f'./{i+1}.json'
        datasets.append(CocoDetection(img_folder, processor, f'{len(splits)}.json'))
    else:
        datasets = [CocoDetection(img_folder, annotations_file, processor)]

    collate_fn = get_collator(processor)

    return {
        'dataset': datasets,
        'dataloader': [DataLoader(i, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, num_workers=num_workers) for i in datasets],
        'examples': datasets[0][0][1]
    }