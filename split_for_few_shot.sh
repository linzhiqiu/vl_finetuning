#!/bin/bash

DATASET=imagenet
# DATASET=oxford_flowers

for SHOTS in 1 2 4 8 16
do 
    for SEED in 1 2 3
    do
        python split_for_few_shot.py \
        --dataset-config-file config/datasets/${DATASET}.yaml \
        --few-shot-config-file config/few_shot/shot_${SHOTS}.yaml \
        SEED ${SEED}
    done
done