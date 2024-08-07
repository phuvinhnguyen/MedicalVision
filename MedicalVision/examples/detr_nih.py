from ..trainer.detection import DetectionTrainer
from ..dataset.detection.CocoDataset import get_loader
from ..models.detection.detr import Detr
from ..utils.visualize import test_and_visualize_model
from transformers import DetrImageProcessor

def run(hf_id,
        token=None,
        image_path='/kaggle/input/nih-dataset-coco-detection/mydataset/images',
        annotations_file='/kaggle/input/nih-dataset-coco-detection/mydataset/annotations.json',
        max_epochs=100,
        lr=1e-4,
        ):
    id2label = {0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion", 3: "Infiltrate", 4: "Mass", 5: "Nodule", 6: "Pneumonia", 7: "Pneumothorax"}
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    dataset = get_loader(image_path, processor, annotations_file=annotations_file)
    model = Detr(dataset['dataloader'][0], dataset['dataloader'][1], lr=lr, id2label=id2label)

    trainer = DetectionTrainer(model, processor, max_epochs=max_epochs)
    test_and_visualize_model(dataset['dataset'][2], model.model, processor, image_idx=1, image_dir=image_path)
    
    trainer.fit()
    trainer.test(dataset['dataloader'][2], dataset['dataset'][2])

    model.model.push_to_hub(hf_id, token=token)
    processor.push_to_hub(hf_id, token=token)

    test_and_visualize_model(dataset['dataset'][2], model.model, processor, image_idx=1, image_dir=image_path)

    return model
