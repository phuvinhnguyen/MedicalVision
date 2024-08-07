from ..Setup.Detection.Setup import Runner


def run(hf_id,
        token=None,
        max_epochs=100,
        lr=1e-4,
        dataset={},
        do_not_train=False
        ):
    runner = Runner(
        'facebook/detr-resnet-50',
        hf_repo_id=hf_id,
        token=token, dataset=('NIH', dataset),
        num_labels=8,
        max_epochs=max_epochs,
        lr=lr
    )

    if do_not_train:
        result, labels = runner.run_example()
        return None, result, labels
    else:
        output = runner.fit()
        return output
