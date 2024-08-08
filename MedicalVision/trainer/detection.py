from pytorch_lightning import Trainer
import torch
from coco_eval import CocoEvaluator
from contextlib import redirect_stdout
import io
from ..utils.tracker.lightning import MetricTracker
from tqdm.notebook import tqdm

def get_evaluation_summary(evaluator):
    string_io = io.StringIO()
    with redirect_stdout(string_io):
        evaluator.summarize()
    summary = string_io.getvalue()
    string_io.close()

    return summary

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

class DetectionTrainer:
    def __init__(self,
                 model,
                 processor = None,
                 max_epochs: int=10,
                 save_path: str=None,
                 device='cuda',
                 local_tracker=True,
                 ) -> None:
        self.tracker = []
        if local_tracker:
            self.tracker.append(MetricTracker())
        self.trainer = Trainer(
            max_epochs=max_epochs,
            callbacks=self.tracker,
            log_every_n_steps=10,
        )
        self.device = device
        self.model = model.to(device)
        self.save_path = save_path
        self.processor = processor

    def fit(self):
        self.trainer.fit(model=self.model, ckpt_path=self.save_path)

    def push_to_hub(self, hub_id, token):
        self.model.push_to_hub(hf_repo_id=hub_id, token=token)
        if self.processor is not None:
            self.processor.push_to_hub(hf_repo_id=hub_id, token=token)

    def test(self, test_dataloader, test_dataset, visualize_first_sample=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        evaluator = CocoEvaluator(coco_gt=test_dataset.coco, iou_types=["bbox"])

        print("Running evaluation...")
        for idx, batch in enumerate(tqdm(test_dataloader)):
            pixel_values = batch['pixel_values'].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

                orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
                results = self.processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)

                predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
                predictions = prepare_for_coco_detection(predictions)
                evaluator.update(predictions)

        evaluator.synchronize_between_processes()
        evaluator.accumulate()

        return get_evaluation_summary(evaluator)