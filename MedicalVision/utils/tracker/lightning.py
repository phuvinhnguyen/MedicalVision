from pytorch_lightning import Callback

class MetricTracker(Callback):
    def __init__(self):
        self.validation = []
        self.training = []

    def on_validation_batch_end(self, trainer, module, outputs, **kwargs):
        vacc = outputs['val_acc']
        self.validation.append(vacc)

    def on_validation_epoch_end(self, trainer, module, **kwargs):
        elogs = trainer.logged_metrics
        self.training.append(elogs)
