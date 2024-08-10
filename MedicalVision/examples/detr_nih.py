from ..dataset.detection.CocoDataset import get_loader
from ..models.detection.detr import Detr
from transformers import DetrImageProcessor
import torch
from ..trainer.detection.detection import DetectionTrainer
from ..utils.uploadReadme import replace_readme_in_hf_repo

def run(hf_id,
        token=None,
        pretrained_model_name_or_path='facebook/detr-resnet-50',
        train_image_path='/kaggle/input/nih-detection-dataset/train',
        train_annotations_file='/kaggle/input/nih-detection-dataset/annotations/train.json',
        valid_image_path='/kaggle/input/nih-detection-dataset/val',
        valid_annotations_file='/kaggle/input/nih-detection-dataset/annotations/val.json',
        test_image_path='/kaggle/input/nih-detection-dataset/val',
        test_annotations_file='/kaggle/input/nih-detection-dataset/annotations/val.json',
        max_epochs=100,
        batch_size=32,
        lr=1e-4,
        dropout_rate=0.1,
        weight_decay=1e-4,
        pull_revision='no_timm',
        train_full=True,
        push_revision=None,
        ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = DetrImageProcessor.from_pretrained(pretrained_model_name_or_path)
    train_dataset = get_loader(train_image_path, processor, annotations_file=train_annotations_file, batch_size=batch_size)
    valid_dataset = get_loader(valid_image_path, processor, annotations_file=valid_annotations_file, batch_size=batch_size)
    test_dataset = get_loader(test_image_path, processor, annotations_file=test_annotations_file, batch_size=batch_size)
    model = Detr(
        train_dataset['dataloader'][0],
        valid_dataset['dataloader'][0],
        lr=lr,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        id2label={k:v['name'] for k,v in train_dataset['dataset'][0].coco.cats.items()},
        model_name=pretrained_model_name_or_path,
        revision=pull_revision
    )

    # Set all parameters trainable
    if train_full:
        for param in model.parameters():
            param.requires_grad = True

    print(f'''Model state:
- All parameter: {sum(p.numel() for p in model.parameters())}
- Trainable parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}''')

    trainer = DetectionTrainer(model, processor, max_epochs=max_epochs, device=device)

    initial_result = trainer.test(test_dataset['dataloader'][0], test_dataset['dataset'][0])
    trainer.visualize(train_dataset['dataset'][0], image_dir=train_image_path)

    trainer.fit()
    final_result = trainer.test(test_dataset['dataloader'][0], test_dataset['dataset'][0])

    validation_tracker_epoch = ''
    if trainer.trackers:
        validation_tracker_epoch = '\n'.join([str(i) for i in trainer.trackers[0].validation_epoch_end])

    commit_message = f'''---
library_name: transformers
tags: []
---

## Original result
```
{initial_result}```

## After training result
```
{final_result}```

## Config
- dataset: NIH
- original model: {pretrained_model_name_or_path}
- lr: {lr}
- dropout_rate: {dropout_rate}
- weight_decay: {weight_decay}
- max_epochs: {max_epochs}

## Logging
### Training process
```
{validation_tracker_epoch}
```
'''
    with open('./README.md', 'w') as wf:
        wf.write(commit_message)

    trainer.push_to_hub(hf_id, token, revision=push_revision)

    replace_readme_in_hf_repo('./README.md', hf_id, token)

    trainer.visualize(train_dataset['dataset'][0], image_dir=train_image_path)
    return model
