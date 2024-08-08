from pytorch_lightning import Callback

class MetricTracker(Callback):
    def __init__(self):
        self.validation_batch_end = []
        self.validation_epoch_end = []
        self.training_batch_end = []
        self.training_epoch_end = []

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        vacc = outputs['val_acc']
        self.validation_batch_end.append(vacc)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        vacc = outputs['val_acc']
        self.training_batch_end.append(vacc)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        elogs = trainer.logged_metrics
        self.validation_epoch_end.append(elogs)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        elogs = trainer.logged_metrics
        self.training_epoch_end.append(elogs)


