import pytorch_lightning as pl
import torch

class lightning_detection(pl.LightningModule):
    """
    Lightning Module for DETR object detection.
    """

    def __init__(self,
                 train_dataloader,
                 eval_dataloader,
                 id2label=None,
                 lr=5e-5,
                 lr_backbone=None,
                 weight_decay=1e-4,
                 dropout_rate=0.1,
                 model_name='facebook/detr-resnet-50',
                 revision="no_timm",
                 ):
        super().__init__()
        self.train_data = train_dataloader
        self.val_data = eval_dataloader
        self.id2label = id2label
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate

        self._init_model(model_name, revision)
    
    def push_to_hub(self, hf_repo_id, token, revision=None, **kwargs):
        self.model.push_to_hub(hf_repo_id, token=token, revision=revision, **kwargs)

    def _init_model(self, model_name, revision):
        pass

    def forward(self, pixel_values, pixel_mask):
        """
        Forward pass of the model.
        """
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        """
        Common step for training and validation.
        """
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        return outputs.loss, outputs.loss_dict

    def training_step(self, batch, batch_idx):
        """
        Training step for the LightningModule.
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the LightningModule.
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v.item())
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers for the LightningModule.
        """
        params = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        """
        Training dataloader for the LightningModule.
        """
        return self.train_data

    def val_dataloader(self):
        """
        Validation dataloader for the LightningModule.
        """
        return self.val_data
