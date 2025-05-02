#!/bin/bash

# Experiment and Data Configuration
EXPNAME="MSMD_model"
# DATA_ROOT="/mnt/f/chrome_downloads/processed_data"
DATA_ROOT="/data/celebv-text/processed_data"
# Model and Training Configuration
SCHEDULER="Warmup"
AUDIO_MODEL="hubert"
STYLE_ENC_MODEL_STYLE="vae2"
GENERATOR_MODEL_STYLE="MSMD"
TRAINING_LOSS_STYLE="MSMD"
DATASET_TYPE='ravdess+celebv-text-medium'
NUM_WORKERS=2
USE_INDICATOR="--use_indicator"
D_STYLE=256
L_KL_DIV=1E-7
L_SMOOTH=1E1 
USE_CROSS_STYLE="--use_cross_style"
USE_VERTEX_SPACE="--use_vertex_space"
MAX_ITER=2000000
NUM_OF_BASIS=4
BATCH_SIZE=16
PROB_CROSS_STYLE=0.5
# Execute Python Script with Arguments
python training_script.py \
    --exp_name $EXPNAME \
    --data_root $DATA_ROOT \
    $USE_INDICATOR \
    $USE_CROSS_STYLE \
    --batch_size $BATCH_SIZE \
    --num_of_basis $NUM_OF_BASIS \
    --scheduler $SCHEDULER \
    --audio_model $AUDIO_MODEL \
    --style_enc_model_style $STYLE_ENC_MODEL_STYLE \
    --generator_model_style $GENERATOR_MODEL_STYLE \
    --training_loss_style $TRAINING_LOSS_STYLE \
    --dataset_type $DATASET_TYPE \
    --num_workers $NUM_WORKERS \
    --d_style $D_STYLE \
    --l_kl_div $L_KL_DIV \
    --l_smooth $L_SMOOTH \
    --max_iter $MAX_ITER \
    --prob_cross_style $PROB_CROSS_STYLE