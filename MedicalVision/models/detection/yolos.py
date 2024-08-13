from transformers import YolosForObjectDetection
from .lightning import lightning_detection
from ...utils.model import change_dropout_rate

class Yolos(lightning_detection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _init_model(self, model_name, revision):
        self.model = YolosForObjectDetection.from_pretrained(
            model_name,
            revision=revision,
            num_labels=len(self.id2label),
            ignore_mismatched_sizes=True,
        ).to(self.device)

        change_dropout_rate(self.model, self.dropout_rate)

    def forward(self, pixel_values, pixel_mask=None):
        """
        Forward pass of the model.
        """
        return self.model(pixel_values=pixel_values)
    
    def common_step(self, batch, batch_idx):
        """
        Common step for training and validation.
        """
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        return outputs.loss, outputs.loss_dict
        