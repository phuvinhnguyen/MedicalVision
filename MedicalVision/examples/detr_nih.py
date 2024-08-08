from ..trainer.detection import DetectionTrainer
from ..dataset.detection.CocoDataset import get_loader
from ..models.detection.detr import Detr
from ..utils.visualize import test_and_visualize_model
from transformers import DetrImageProcessor
import torch
from ..utils.uploadReadme import replace_readme_in_hf_repo

def run(hf_id,
        token=None,
        pretrained_model_name_or_path='facebook/detr-resnet-50',
        image_path='/kaggle/input/nih-dataset-coco-detection/mydataset/images',
        annotations_file='/kaggle/input/nih-dataset-coco-detection/mydataset/annotations.json',
        max_epochs=100,
        batch_size=32,
        lr=1e-4,
        index_test_dataset=2,
        revision='no_timm',
        train_full=True,
        ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2label = {0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion", 3: "Infiltrate", 4: "Mass", 5: "Nodule", 6: "Pneumonia", 7: "Pneumothorax"}
    processor = DetrImageProcessor.from_pretrained(pretrained_model_name_or_path)
    dataset = get_loader(image_path, processor, annotations_file=annotations_file, batch_size=batch_size)
    model = Detr(
        dataset['dataloader'][0],
        dataset['dataloader'][1],
        lr=lr,
        id2label=id2label,
        model_name=pretrained_model_name_or_path,
        revision=revision
        )

    # Set all parameters trainable
    if train_full:
        for param in model.parameters():
            param.requires_grad = True

    print(f'''Model state:
- All parameter: {sum(p.numel() for p in model.parameters())}
- Trainable parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}''')

    trainer = DetectionTrainer(model, processor, max_epochs=max_epochs, device=device)
    initial_result = trainer.test(dataset['dataloader'][index_test_dataset], dataset['dataset'][index_test_dataset])
    test_and_visualize_model(dataset['dataset'][index_test_dataset], model.model, processor, image_idx=1, image_dir=image_path, device=device)

    trainer.fit()
    final_result = trainer.test(dataset['dataloader'][index_test_dataset], dataset['dataset'][index_test_dataset])

    validation_tracker_epoch = ''
    training_tracker_epoch = ''
    if trainer.tracker:
        validation_tracker_epoch = '\n'.join([str(i) for i in trainer.tracker[0].validation_epoch_end])
        training_tracker_epoch = '\n' + '\n'.join([str(i) for i in trainer.tracker[0].training_epoch_end])

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
- max_epochs: {max_epochs}

## Logging

### Training process
```
{training_tracker_epoch}
```

### Validation process
```
{validation_tracker_epoch}
```
'''
    with open('./README.md', 'w') as wf:
        wf.write(commit_message)

    model.model.push_to_hub(hf_id, token=token, commit_message=commit_message)
    processor.push_to_hub(hf_id, token=token)

    replace_readme_in_hf_repo('./README.md', hf_id, token)

    test_and_visualize_model(dataset['dataset'][index_test_dataset], model.model, processor, image_idx=1, image_dir=image_path, device=device)

    return model
