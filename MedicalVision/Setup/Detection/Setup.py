from torch.utils.data import DataLoader
from .ModelWrapper import GeneralDetectionModel
from pytorch_lightning import Trainer
from ...Dataset.Detection import EXISTING_DATASET
from transformers import AutoModelForObjectDetection, AutoProcessor
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes, model=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        if model is not None:
            label = model.config.id2label[label]
        else:
            label = int(label)
        text = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def get_collator(model_name_or_path):
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch

    return collate_fn, processor


class Runner:
    def __init__(self,
                 model_name_or_path,
                 hf_repo_id,
                 token,
                 dataset,
                 lr=5e-5,
                 max_epochs=3,
                 batch_size=4,
                 num_labels=1000,
                 max_steps=None,) -> None:
        self.train_data, self.val_data, self.test_data = \
            EXISTING_DATASET[dataset[0]](**dataset[1])

        processor, self.model_processor = get_collator(model_name_or_path)

        self.train_dataset = DataLoader(
            self.train_data, collate_fn=processor, batch_size=batch_size)
        self.val_dataset = DataLoader(
            self.val_data, collate_fn=processor, batch_size=batch_size)
        self.test_dataset = DataLoader(
            self.test_data, collate_fn=processor, batch_size=batch_size)

        self.model = GeneralDetectionModel(
            AutoModelForObjectDetection.from_pretrained(
                model_name_or_path,
                revision="no_timm",
                num_labels=num_labels,
                ignore_mismatched_sizes=True),
            self.train_dataset,
            self.val_dataset,
            lr=lr,
            processor=self.model_processor
        )

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.token = token
        self.hf_repo_id = hf_repo_id

    def fit(self):
        if self.max_steps is None:
            trainer = Trainer(
                max_epochs=self.max_epochs,
            )
        else:
            trainer = Trainer(
                max_steps=self.max_steps
            )

        trainer.test(self.model, self.test_dataset)
        before_output = self.model.compute_and_reset()
        self.run_example()

        trainer.fit(self.model)

        trainer.test(self.model, self.test_dataset)
        after_output = self.model.compute_and_reset()
        self.run_example()

        self.model.model.push_to_hub(self.hf_repo_id, token=self.token)
        self.model_processor.push_to_hub(self.hf_repo_id, token=self.token)

        return trainer, before_output, after_output
    
    def run_example(self, run_set='train', index=0):
        example = None
        if run_set == 'train':
            example = self.train_data[index]
        elif run_set == 'val':
            example = self.val_data[index]
        elif run_set == 'test':
            example = self.test_data[index]

        image, labels = example
        pixel_values = image.unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=None)
            print("Outputs:", outputs.keys())

        results = self.model_processor.post_process_object_detection(outputs,
                                                                target_sizes=[(1024, 1024)],
                                                                threshold=0.9)[0]
        
        plot_results(ToPILImage()(image), results['scores'], results['labels'], results['boxes'])
        print(results)
        print(labels)

        return results, labels