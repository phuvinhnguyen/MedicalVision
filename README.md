# MedicalVision

## Install
```python
pip install git+https://github.com/phuvinhnguyen/MedicalVision.git
```

# Train and Evaluating DETR
```python
from MedicalVision.Examples.DETR_NIH import run

_, before_training_result, after_training_result = run('MedicalVision/test', token='<YOUR TOKEN>', dataset={})

print(before_training_result)

print(after_training_result)
```