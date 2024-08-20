#!/bin/bash

# Usage: ./run_program.sh <hf_id> [other options]

# Assign variables from command-line arguments
HF_ID=$1
TOKEN=$2
PRETRAINED_MODEL=${3:-'hustvl/yolos-tiny'}
TRAIN_IMAGE_PATH=${4:-'/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train'}
TRAIN_ANNOTATIONS_FILE=${5:-'/kaggle/working/coco_dataset.json'}
VALID_IMAGE_PATH=${6:-'/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train'}
VALID_ANNOTATIONS_FILE=${7:-'/kaggle/working/coco_dataset.json'}
TEST_IMAGE_PATH=${8:-'/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train'}
TEST_ANNOTATIONS_FILE=${9:-'/kaggle/working/coco_dataset.json'}
MAX_EPOCHS=${10:-3}
BATCH_SIZE=${11:-4}
LR=${12:-1e-4}
DROPOUT_RATE=${13:-0.1}
WEIGHT_DECAY=${14:-1e-4}
PULL_REVISION=${15:-''}
DO_TRAIN=${16:-'--do_train'}
PUSH_TO_HUB=${17:-''}
TRAIN_FULL=${18:-'--train_full'}
PUSH_REVISION=${19:-''}
EXAMPLE_PATH=${20:-'/kaggle/working/example.png'}
VISUALIZE_THRESHOLD=${21:-0.1}
JUST_VISUAL=${22:-''}
VISUALIZE_IDX=${23:-1}

# Run the Python program
python3 MedicalVision/MedicalVision/examples/yolos_vinxray.py \
    --hf_id $HF_ID \
    --token $TOKEN \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --train_image_path $TRAIN_IMAGE_PATH \
    --train_annotations_file $TRAIN_ANNOTATIONS_FILE \
    --valid_image_path $VALID_IMAGE_PATH \
    --valid_annotations_file $VALID_ANNOTATIONS_FILE \
    --test_image_path $TEST_IMAGE_PATH \
    --test_annotations_file $TEST_ANNOTATIONS_FILE \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --dropout_rate $DROPOUT_RATE \
    --weight_decay $WEIGHT_DECAY \
    --pull_revision $PULL_REVISION \
    $DO_TRAIN \
    $PUSH_TO_HUB \
    $TRAIN_FULL \
    --push_revision $PUSH_REVISION \
    --example_path $EXAMPLE_PATH \
    --visualize_threshold $VISUALIZE_THRESHOLD \
    $JUST_VISUAL \
    --visualize_idx $VISUALIZE_IDX
