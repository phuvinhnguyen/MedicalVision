from ..Setup.Detection.Setup import Runner


def run(hf_id, token=None, dataset={}):
    runner = Runner(
        'facebook/detr-resnet-50',
        hf_repo_id=hf_id,
        token=token, dataset=('NIH', dataset),
        num_labels=8
    )

    output = runner.fit()
    
    return output
