#!/bin/bash
export SIGN_PREFIX="lower"
export SIGN_MASK="lower.png"
export EPOCH=299
export OUT_FILENAME="lower299"

python apply_noise_to_bigger_image.py \
    --big_image ./misc/uw17-cropped.png \
    --model_path ./optimization_output/${SIGN_PREFIX}/model/${SIGN_PREFIX}-${EPOCH} \
    --output_path ./misc/${OUT_FILENAME}.png \
    --attack_mask ./masks/octagon.png

