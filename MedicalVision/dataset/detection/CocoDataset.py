import torchvision
import os
from torch.utils.data import DataLoader
from ...utils import split_train_test

def get_collator(processor):
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch
    return collate_fn

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, ann_file=None):
        if ann_file is None:
            ann_file = os.path.join(img_folder, "../annotations.json")
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
    
def get_loader(
    img_folder,
    processor,
    annotations_file=None,
    batch_size=1,
    num_workers=0,
    splits=[.8, .5],
    shuffle=True,
    ):
    if annotations_file is not None:
        datasets = []
        for i, split in enumerate(splits):
            split_train_test(annotations_file, f'{i}.json', f'{i+1}.json', split=split)
            datasets.append(CocoDetection(img_folder, processor, f'{i}.json'))
        datasets.append(CocoDetection(img_folder, processor, f'{len(splits)}.json'))
    else:
        datasets = [CocoDetection(img_folder, processor)]

    collate_fn = get_collator(processor)

    return {
        'dataset': datasets,
        'dataloader': [DataLoader(i, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, num_workers=num_workers) for i in datasets]
    }