#!/bin/bash

fg_threshold=0.40
bg_threshold=0.10
EXPERIMENT_NAME=resnet101

python -m make_affinity_labels \
   --experiment_name ${EXPERIMENT_NAME} \
   --domain train_aug \
   --fg_threshold ${fg_threshold} \
   --bg_threshold ${bg_threshold} \
   --pred_dir ../resim/results/pipeline/resnet101v2/irnet/make_cam/seg-model-249/cam_outputs/

python -m train_affinitynet \
  --architecture ${EXPERIMENT_NAME} \
  --tag AffinityNet@${EXPERIMENT_NAME}@aff_fg=${fg_threshold}_bg=${bg_threshold} \
  --label_name ${EXPERIMENT_NAME}@aff_fg=${fg_threshold}_bg=${bg_threshold}

python -m inference_rw \
  --architecture ${EXPERIMENT_NAME} \
  --model_name AffinityNet@${EXPERIMENT_NAME}@aff_fg=${fg_threshold}_bg=${bg_threshold} \
  --cam_dir ../resim/results/pipeline/resnet101v2/irnet/make_cam/seg-model-249/cam_outputs/ \
  --domain train_aug

python evaluate.py \
  --experiment_name AffinityNet@${EXPERIMENT_NAME}@aff_fg=${fg_threshold}_bg=${bg_threshold}@train@beta=10@exp_times=8@rw \
  --domain train

python -m make_pseudo_labels \
  --experiment_name AffinityNet@${EXPERIMENT_NAME}@aff_fg=${fg_threshold}_bg=${bg_threshold}@train@beta=10@exp_times=8@rw \
  --domain train_aug \
  --threshold 0.20 \
  --crf_iteration 1

python evaluate.py \
  --experiment_name AffinityNet@${EXPERIMENT_NAME}@aff_fg=${fg_threshold}_bg=${bg_threshold}@train@beta=10@exp_times=8@rw@crf=1 \
  --domain train \
  --mode png

python train_segmentation.py \
 --backbone resnet101 \
 --mode fix \
 --use_gn True \
 --tag DeepLabv3+resnet101@${EXPERIMENT_NAME}@aff_fg=${fg_threshold}_bg=${bg_threshold}@Fix@GN \
 --label_name AffinityNet@${EXPERIMENT_NAME}@aff_fg=${fg_threshold}_bg=${bg_threshold}@train@beta=10@exp_times=8@rw@crf=1 \
 --batch_size 16

python inference_segmentation.py \
  --backbone ${EXPERIMENT_NAME} \
  --mode fix \
  --use_gn True \
  --tag DeepLabv3+resnet101@${EXPERIMENT_NAME}@aff_fg=${fg_threshold}_bg=${bg_threshold}@Fix@GN \
  --scale 0.5,1.0,1.5,2.0 \
  --iteration 10

python -m evaluate \
  --experiment_name DeepLabv3+resnet101@${EXPERIMENT_NAME}@aff_fg=${fg_threshold}_bg=${bg_threshold}@Fix@GN@val@scale=0.5,1.0,1.5,2.0@iteration=10 \
  --domain val

