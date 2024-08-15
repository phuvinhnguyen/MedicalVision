from ..dataset.detection.CocoDataset import get_loader
from ..models.detection.detr import Detr
from transformers import DetrImageProcessor
import torch
from ..trainer.detection.detection import DetectionTrainer
from ..utils.uploadReadme import write_file_in_hf_repo
from ..utils.model import set_all_params_to_trainable, model_params

def run(hf_id,
        token=None,
        pretrained_model_name_or_path='facebook/detr-resnet-50',
        train_image_path='/kaggle/working/balloon/train',
        train_annotations_file='/kaggle/working/balloon/train/custom_train.json',
        valid_image_path='/kaggle/working/balloon/val',
        valid_annotations_file='/kaggle/working/balloon/val/custom_val.json',
        test_image_path='/kaggle/working/balloon/val',
        test_annotations_file='/kaggle/working/balloon/val/custom_val.json',
        max_epochs=100,
        batch_size=32,
        lr=1e-4,
        dropout_rate=0.1,
        weight_decay=1e-4,
        pull_revision='no_timm',
        do_train=True,
        push_to_hub=False,
        train_full=True,
        push_revision=None,
        example_path='/kaggle/working/example.png',
        visualize_threshold=0.1,
        just_visual=False,
        ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2label = {0: 'balloon'}

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
        id2label=id2label,
        model_name=pretrained_model_name_or_path,
        revision=pull_revision
    )

    # Set all parameters trainable
    if train_full: set_all_params_to_trainable(model)

    model_params(model)

    trainer = DetectionTrainer(model, processor, max_epochs=max_epochs, device=device)

    if just_visual:
        trainer.visualize(train_dataset['dataset'][0], image_dir=train_image_path, threshold=visualize_threshold)
        return model
    
    initial_result = trainer.test(test_dataset['dataloader'][0], test_dataset['dataset'][0])
    print(initial_result)
    trainer.visualize(train_dataset['dataset'][0], image_dir=train_image_path, threshold=visualize_threshold)

    if do_train:
        trainer.fit()
        final_result = trainer.test(test_dataset['dataloader'][0], test_dataset['dataset'][0])
        trainer.visualize(train_dataset['dataset'][0], image_dir=train_image_path, threshold=visualize_threshold)

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
    - dataset: NIH
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
