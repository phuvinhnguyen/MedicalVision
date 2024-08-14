# MedicalVision

## Install
```python
pip install git+https://github.com/phuvinhnguyen/MedicalVision.git
```

# Train and Evaluating DETR
```python
from MedicalVision.examples.yolos_nih import run

run("MedicalVision/yolos_tiny_30ep",
    token='<YOUR TOKEN>',
    batch_size=16,
    dropout_rate=0.1,
    weight_decay=0.001,
    max_epochs=30,
    pull_revision=None,
    push_revision=None,
    push_to_hub=True
   )
```