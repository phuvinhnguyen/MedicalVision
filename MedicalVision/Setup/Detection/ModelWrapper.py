import pytorch_lightning as pl
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class GeneralDetectionModel(pl.LightningModule):
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 lr=5e-5,
                 lr_backbone=None,
                 weight_decay=0.0,
                 num_labels=1000,
                 processor=None,
                 ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.processor = processor

        # Initialize metrics
        self.map_metric = MeanAveragePrecision()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch.get("pixel_mask", None)
        labels = [{k: v.to(self.device) for k, v in t.items()}
                  for t in batch["labels"]]

        # Forward pass with labels to calculate loss
        outputs = self.model(pixel_values=pixel_values,
                             pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Set up parameter groups for optimization
        param_dicts = [
            {"params": [p for n, p in self.named_parameters(
            ) if "backbone" not in n and p.requires_grad]},
        ]
        if self.lr_backbone is not None:
            param_dicts.append({
                "params": [
                    p for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            })

        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        # Placeholder for training DataLoader
        return self.train_data

    def val_dataloader(self):
        # Placeholder for validation DataLoader
        return self.val_data
    
    def format_predictions(self, predictions):
        # Format predictions to match the expected format for metrics
        formatted_preds = []
        for pred in predictions:
            print(pred)
            boxes = pred['boxes'].cpu()  # Bounding boxes
            scores = pred['scores'].cpu()  # Confidence scores
            labels = pred['labels'].cpu()  # Predicted labels
            formatted_preds.append({"boxes": boxes, "scores": scores, "labels": labels})
        return formatted_preds

    def format_labels(self, labels):
        # Format labels to match the expected format for metrics
        formatted_labels = []
        for label in labels:
            boxes = label['boxes'].cpu()  # Bounding boxes
            labels = label['labels'].cpu()  # Ground truth labels
            formatted_labels.append({"boxes": boxes, "labels": labels})
        return formatted_labels
        
    def test_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch.get("pixel_mask", None)
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        # Forward pass without labels to get predictions
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            results = self.processor.post_process_object_detection(outputs, threshold=0)
            
        formatted_preds = self.format_predictions(results)
        formatted_labels = self.format_labels(labels)

        self.map_metric.update(formatted_preds, formatted_labels)

        return formatted_preds, formatted_labels