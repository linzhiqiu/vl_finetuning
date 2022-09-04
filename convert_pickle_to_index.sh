#!/bin/bash


for DATASET in caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101
do
    for SHOTS in 1 2 4 8 16
    do 
        for SEED in 1 2 3
        do
            python convert_pickle_to_index.py \
            --dataset-config-file config/datasets/${DATASET}.yaml \
            --few-shot-config-file config/few_shot/shot_${SHOTS}.yaml \
            SEED ${SEED}
        done
    done
done