from transformers import Owlv2ForObjectDetection
from .lightning import lightning_detection
from ...utils.model import change_dropout_rate

class OwlV2(lightning_detection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _init_model(self, model_name, revision):
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_name,
            revision=revision,
            num_labels=len(self.id2label),
            ignore_mismatched_sizes=True,
        ).to(self.device)

        change_dropout_rate(self.model, self.dropout_rate)
        