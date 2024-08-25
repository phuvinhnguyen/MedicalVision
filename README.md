# MedicalVision

## Install

```python
pip install git+https://github.com/phuvinhnguyen/MedicalVision.git
```

# 
    Train and Evaluating YOLOS on VinXray

```python
pip install git+https://github.com/phuvinhnguyen/MedicalVision.git
git clone https://github.com/phuvinhnguyen/MedicalVision.git

!python MedicalVision/MedicalVision/examples/yolos_vinxray.py \
    --hf_id MedicalVision/example \
    --token <YOUR_HF_TOKEN> \
    --pretrained_model_name_or_path 'hustvl/yolos-tiny' \
    --max_epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout_rate 0.1 \
    --weight_decay 1e-4 \
    --do_train \
    --push_to_hub \
    --example_path './example.png' \
    --visualize_threshold 0.1 \
    --visualize_idx 1 \
    --wandb_key <YOUR_WANDB_KEY> \
    --wandb_project 'ExampleProject' \
    --checkpoint_path './yolos_vinxray_ckpt.pt'

```

# Get Gradcam of YOLOS

```python
from MedicalVision.models.detection.yolos import Yolos, YolosForObjectDetection
from PIL import Image

image = Image.open('./<some_image>.jpeg')

model = Yolos(None, None, model_name='hustvl/yolos-tiny', revision=None, id2label=[1]*92)
model.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

attention_map = model.run(image, threshold=0.1, with_bbox=True, with_image=True, with_smooth=True)

attention_map # Appear as a dict of categories and images
```
