#!/bin/bash

# Usage: ./run_model.sh <hf_id> [other options]

# Assign variables from command-line arguments
HF_ID=$1
TOKEN=${2:-""}
PRETRAINED_MODEL_NAME_OR_PATH=${3:-'facebook/detr-resnet-50'}
TRAIN_IMAGE_PATH=${4:-'/kaggle/input/nih-detection-dataset/train'}
TRAIN_ANNOTATIONS_FILE=${5:-'/kaggle/input/nih-detection-dataset/annotations/train.json'}
VALID_IMAGE_PATH=${6:-'/kaggle/input/nih-detection-dataset/val'}
VALID_ANNOTATIONS_FILE=${7:-'/kaggle/input/nih-detection-dataset/annotations/val.json'}
TEST_IMAGE_PATH=${8:-'/kaggle/input/nih-detection-dataset/val'}
TEST_ANNOTATIONS_FILE=${9:-'/kaggle/input/nih-detection-dataset/annotations/val.json'}
MAX_EPOCHS=${10:-100}
BATCH_SIZE=${11:-32}
LR=${12:-1e-4}
DROPOUT_RATE=${13:-0.1}
WEIGHT_DECAY=${14:-1e-4}
PULL_REVISION=${15:-'no_timm'}
DO_TRAIN=${16:-'True'}
PUSH_TO_HUB=${17:-'False'}
TRAIN_FULL=${18:-'True'}
PUSH_REVISION=${19:-""}
EXAMPLE_PATH=${20:-'/kaggle/working/example.png'}
VISUALIZE_THRESHOLD=${21:-0.1}
VISUALIZE_IDX=${22:-1}
JUST_VISUAL=${23:-'False'}

# Run the Python program
python3 your_python_script.py \
    --hf_id "$HF_ID" \
    --token "$TOKEN" \
    --pretrained_model_name_or_path "$PRETRAINED_MODEL_NAME_OR_PATH" \
    --train_image_path "$TRAIN_IMAGE_PATH" \
    --train_annotations_file "$TRAIN_ANNOTATIONS_FILE" \
    --valid_image_path "$VALID_IMAGE_PATH" \
    --valid_annotations_file "$VALID_ANNOTATIONS_FILE" \
    --test_image_path "$TEST_IMAGE_PATH" \
    --test_annotations_file "$TEST_ANNOTATIONS_FILE" \
    --max_epochs "$MAX_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --dropout_rate "$DROPOUT_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --pull_revision "$PULL_REVISION" \
    --do_train "$DO_TRAIN" \
    --push_to_hub "$PUSH_TO_HUB" \
    --train_full "$TRAIN_FULL" \
    --push_revision "$PUSH_REVISION" \
    --example_path "$EXAMPLE_PATH" \
    --visualize_threshold "$VISUALIZE_THRESHOLD" \
    --visualize_idx "$VISUALIZE_IDX" \
    --just_visual "$JUST_VISUAL"