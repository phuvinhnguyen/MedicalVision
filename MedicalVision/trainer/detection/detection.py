from pytorch_lightning import Trainer
from .visualize import plot_from_dataset
from .tracker import MetricTracker
from .evaluation import *
from ..abstract import AbstractTrainer
from lightning.pytorch.loggers import WandbLogger
import wandb

class DetectionTrainer(AbstractTrainer):
    def __init__(self,
                 model,
                 processor = None,
                 max_epochs: int=10,
                 save_path: str=None,
                 device='cuda',
                 ) -> None:
        super().__init__(model, processor, save_path, device, trackers=[MetricTracker()])
        self.max_epochs = max_epochs

    def fit(self, wandb_key=None, wandb_project='XrayImageDetection'):
        if wandb_key is not None:
            wandb.login(key=wandb_key)
            trainer = Trainer(max_epochs=self.max_epochs, callbacks=self.trackers, logger=WandbLogger(project=wandb_project))
        else:
            trainer = Trainer(max_epochs=self.max_epochs, callbacks=self.trackers)
        trainer.fit(model=self.model) #, ckpt_path=self.save_path)

    def test(self, test_dataloader, test_dataset):
        return evaluate_model(
            self.model,
            self.processor,
            test_dataloader,
            test_dataset,
            device=self.device
        )
    
    def visualize(self, val_dataset, image_idx=1, image_dir=None, threshold=0.2):
        plot_from_dataset(
            self.model,
            val_dataset,
            self.processor,
            idx=image_idx,
            threshold=threshold,
            device=self.model.device,
        )