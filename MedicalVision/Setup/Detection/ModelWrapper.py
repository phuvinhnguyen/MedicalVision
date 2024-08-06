import pytorch_lightning as pl
import torch


class GeneralDetectionModel(pl.LightningModule):
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 lr=5e-5,
                 lr_backbone=None,
                 weight_decay=0.0,
                 id2label=None
                 ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.id2label = id2label
        self.train_data = train_dataloader
        self.val_data = val_dataloader

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
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

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
