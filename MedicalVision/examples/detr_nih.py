from ..trainer.detection import DetectionTrainer
from ..dataset.detection.CocoDataset import get_loader
from ..models.detection.detr import Detr
from ..utils.visualize import test_and_visualize_model
from transformers import DetrImageProcessor
import torch

def run(hf_id,
        token=None,
        pretrained_model_name_or_path='facebook/detr-resnet-50',
        image_path='/kaggle/input/nih-dataset-coco-detection/mydataset/images',
        annotations_file='/kaggle/input/nih-dataset-coco-detection/mydataset/annotations.json',
        max_epochs=100,
        lr=1e-4,
        index_test_dataset=2,
        revision='no_timm',
        ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2label = {0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion", 3: "Infiltrate", 4: "Mass", 5: "Nodule", 6: "Pneumonia", 7: "Pneumothorax"}
    processor = DetrImageProcessor.from_pretrained(pretrained_model_name_or_path)
    dataset = get_loader(image_path, processor, annotations_file=annotations_file)
    model = Detr(dataset['dataloader'][0], dataset['dataloader'][1], lr=lr, id2label=id2label, model_name=pretrained_model_name_or_path, revision=revision)

    trainer = DetectionTrainer(model, processor, max_epochs=max_epochs, device=device)
    trainer.test(dataset['dataloader'][index_test_dataset], dataset['dataset'][index_test_dataset])
    test_and_visualize_model(dataset['dataset'][index_test_dataset], model.model, processor, image_idx=1, image_dir=image_path, device=device)

    trainer.fit()
    trainer.test(dataset['dataloader'][index_test_dataset], dataset['dataset'][index_test_dataset])

    model.model.push_to_hub(hf_id, token=token)
    processor.push_to_hub(hf_id, token=token)

    test_and_visualize_model(dataset['dataset'][index_test_dataset], model.model, processor, image_idx=1, image_dir=image_path, device=device)

    return model
