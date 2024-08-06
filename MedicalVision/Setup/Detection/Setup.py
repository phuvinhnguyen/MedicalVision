from torch.utils.data import DataLoader
from ModelWrapper import GeneralDetectionModel
from pytorch_lightning import Trainer
from ...Dataset.Detection import EXISTING_DATASET
from transformers import AutoModelForObjectDetection, AutoProcessor


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
        self.train_dataset, self.val_dataset, self.test_dataset = \
            EXISTING_DATASET[dataset[0]](**dataset[1])

        processor, self.model_processor = get_collator(model_name_or_path)

        self.train_dataset = DataLoader(
            self.train_dataset, collate_fn=processor, batch_size=batch_size)
        self.val_dataset = DataLoader(
            self.val_dataset, collate_fn=processor, batch_size=batch_size)
        self.test_dataset = DataLoader(
            self.test_dataset, collate_fn=processor, batch_size=batch_size)

        self.model = GeneralDetectionModel(
            AutoModelForObjectDetection.from_pretrained(
                model_name_or_path,
                revision="no_timm",
                num_labels=num_labels,
                ignore_mismatched_sizes=True),
            self.train_dataset,
            self.val_dataset,
            lr=lr,
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

        trainer.fit(self.model)

        self.model.model.push_to_hub(self.hf_repo_id, token=self.token)
        self.model_processor.push_to_hub(self.hf_repo_id, token=self.token)
