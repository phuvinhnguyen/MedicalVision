from MedicalVision.dataset.detection.CocoDataset import get_loader
from MedicalVision.models.detection.yolos import Yolos
from transformers import YolosImageProcessor
import torch
from MedicalVision.trainer.detection.detection import DetectionTrainer
from MedicalVision.utils.uploadReadme import write_file_in_hf_repo
from MedicalVision.utils.model import set_all_params_to_trainable, model_params
import argparse

def run(hf_id,
        token=None,
        pretrained_model_name_or_path='hustvl/yolos-tiny',
        train_image_path='/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train',
        train_annotations_file='train.json',
        valid_image_path='/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train',
        valid_annotations_file='train.json',
        test_image_path='/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test',
        test_annotations_file='test.json',
        max_epochs=100,
        batch_size=32,
        lr=1e-4,
        dropout_rate=0.1,
        weight_decay=1e-4,
        pull_revision=None,
        do_train=True,
        push_to_hub=False,
        train_full=True,
        push_revision=None,
        example_path='./example.png',
        visualize_threshold=0.1,
        just_visual=False,
        visualize_idx=1,
        checkpoint_path='./yolos_vin_ckpt.pt',
        wandb_key=None,
        wandb_project='XrayImageDetection',
        ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Running device: {device}')

    processor = YolosImageProcessor.from_pretrained(pretrained_model_name_or_path)
    train_dataset = get_loader(train_image_path, processor, annotations_file=train_annotations_file, batch_size=batch_size)
    valid_dataset = get_loader(valid_image_path, processor, annotations_file=valid_annotations_file, batch_size=batch_size)
    test_dataset = get_loader(test_image_path, processor, annotations_file=test_annotations_file, batch_size=batch_size)
    
    model = Yolos(
        train_dataset['dataloader'][0],
        valid_dataset['dataloader'][0],
        lr=lr,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        id2label={k-1:v['name'] for k,v in train_dataset['dataset'][0].coco.cats.items()},
        model_name=pretrained_model_name_or_path,
        revision=pull_revision
    )

    # Set all parameters trainable
    if train_full: set_all_params_to_trainable(model)

    model_params(model)

    trainer = DetectionTrainer(model, processor, max_epochs=max_epochs, device=device, save_path=checkpoint_path)

    if just_visual:
        trainer.visualize(train_dataset['dataset'][0], image_idx=visualize_idx, image_dir=train_image_path, threshold=visualize_threshold)
        return model

    initial_result = 'Not provided\n' # trainer.test(test_dataset['dataloader'][0], test_dataset['dataset'][0])
    print(initial_result)

    if do_train:
        print('Begin training...')
        trainer.fit(wandb_key=wandb_key, wandb_project=wandb_project)
        final_result = trainer.test(test_dataset['dataloader'][0], test_dataset['dataset'][0])
        trainer.visualize(train_dataset['dataset'][0], image_idx=visualize_idx, image_dir=train_image_path, threshold=visualize_threshold)

        validation_tracker_epoch = ''
        if trainer.trackers: validation_tracker_epoch = '\n'.join([str(i) for i in trainer.trackers[0].validation_epoch_end])

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
- dataset: VinXray
- original model: {pretrained_model_name_or_path}
- lr: {lr}
- dropout_rate: {dropout_rate}
- weight_decay: {weight_decay}
- max_epochs: {max_epochs}
- train samples: {len(train_dataset['dataset'][0])}

## Logging
### Training process
```
{validation_tracker_epoch}
```

## Examples
{train_dataset['examples']}

![Example](./example.png)
'''
        with open('./README.md', 'w') as wf:
            wf.write(commit_message)

        if push_to_hub:
            trainer.push_to_hub(hf_id, token, revision=push_revision)
            write_file_in_hf_repo('./README.md', hf_id, token, revision=push_revision)
            write_file_in_hf_repo(example_path, hf_id, token, revision=push_revision, desfilename='example.png')

    return model

def main():
    parser = argparse.ArgumentParser(description="Run the YOLOS training pipeline")

    parser.add_argument('--hf_id', required=True, help="Hugging Face model ID")
    parser.add_argument('--token', default=None, help="Hugging Face token")
    parser.add_argument('--pretrained_model_name_or_path', default='hustvl/yolos-tiny', help="Pretrained model path or name")
    parser.add_argument('--train_image_path', default='/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train', help="Path to training images")
    parser.add_argument('--train_annotations_file', default='/kaggle/working/MedicalVision/resources/VinXray_train.json', help="Path to training annotations file")
    parser.add_argument('--valid_image_path', default='/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train', help="Path to validation images")
    parser.add_argument('--valid_annotations_file', default='/kaggle/working/MedicalVision/resources/VinXray_val.json', help="Path to validation annotations file")
    parser.add_argument('--test_image_path', default='/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train', help="Path to test images")
    parser.add_argument('--test_annotations_file', default='/kaggle/working/MedicalVision/resources/VinXray_val.json', help="Path to test annotations file")
    parser.add_argument('--max_epochs', type=int, default=100, help="Maximum number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--pull_revision', default=None, help="Revision to pull from Hugging Face")
    parser.add_argument('--do_train', action='store_true', help="Flag to train the model")
    parser.add_argument('--push_to_hub', action='store_true', help="Flag to push the model to Hugging Face Hub")
    parser.add_argument('--train_full', action='store_true', help="Flag to train the full model")
    parser.add_argument('--push_revision', default=None, help="Revision to push to Hugging Face")
    parser.add_argument('--example_path', default='./example.png', help="Path to example image")
    parser.add_argument('--visualize_threshold', type=float, default=0.1, help="Threshold for visualization")
    parser.add_argument('--just_visual', action='store_true', help="Flag to only visualize without training")
    parser.add_argument('--visualize_idx', type=int, default=1, help="Index for visualization")

    args = parser.parse_args()

    run(hf_id=args.hf_id,
        token=args.token,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        train_image_path=args.train_image_path,
        train_annotations_file=args.train_annotations_file,
        valid_image_path=args.valid_image_path,
        valid_annotations_file=args.valid_annotations_file,
        test_image_path=args.test_image_path,
        test_annotations_file=args.test_annotations_file,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout_rate=args.dropout_rate,
        weight_decay=args.weight_decay,
        pull_revision=args.pull_revision,
        do_train=args.do_train,
        push_to_hub=args.push_to_hub,
        train_full=args.train_full,
        push_revision=args.push_revision,
        example_path=args.example_path,
        visualize_threshold=args.visualize_threshold,
        just_visual=args.just_visual,
        visualize_idx=args.visualize_idx,
        )

if __name__ == "__main__":
    main()