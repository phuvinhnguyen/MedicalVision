from ..Setup.Detection.Setup import Runner


def run(hf_id, token=None):
    runner = Runner(
        'facebook/detr-resnet-50',
        hf_repo_id=hf_id,
        token=token, dataset=('NIH',),
        num_labels=8
    )

    runner.fit()
