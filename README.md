# MedicalVision

## Install
```python
pip install git+https://github.com/phuvinhnguyen/MedicalVision.git
```

# Train and Evaluating YOLOS on VinXray
```python
pip install git+https://github.com/phuvinhnguyen/MedicalVision.git
git clone https://github.com/phuvinhnguyen/MedicalVision.git

python MedicalVision/MedicalVision/examples/yolos_vinxray.py \
--hf_id MedicalVision/test_model \
--token <YOUR TOKEN> \
--batch_size 32 \
--max_epochs 100 \
--push_to_hub \
--do_train \
--train_full \
--example_path './example.png' \
--pretrained_model_name_or_path 'hustvl/yolos-tiny' \
--train_image_path <TRAIN_IMG_PATH> \
--valid_image_path <VALID_IMG_PATH> \
--test_image_path <TEST_IMG_PATH> \
```

# Get Gradcam of YOLOS
```python
from MedicalVision.models.detection.yolos import Yolos, YolosForObjectDetection
from PIL import Image

image = Image.open('./<some_image>.jpeg')

model = Yolos(None, None, model_name='hustvl/yolos-tiny', revision=None, id2label=[1]*92)
model.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

attention_map = model.run(image, threshold=0.1)

attention_map # Appear as a dict of categories and images
```