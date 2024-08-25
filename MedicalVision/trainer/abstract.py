class AbstractTrainer:
    def __init__(self,
                 model,
                 processor=None,
                 save_path=None,
                 device='cuda',
                 trackers=[]
                 ) -> None:
        self.model = model
        self.processor = processor
        self.save_path = save_path
        self.device = device
        self.trackers = trackers

    def push_to_hub(self, hf_repo_id, token, revision=None):
        self.model.push_to_hub(hf_repo_id, token=token, revision=revision)
        if self.processor is not None:
            self.processor.push_to_hub(hf_repo_id, token=token, revision=revision)

    