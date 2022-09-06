#!/bin/bash

declare -a FEATURES=("rn50_view_1_ccrop_template_default" "rn50_view_1_ccrop_template_extra" 
                     "rn50_view_100_rcrop_template_default" "rn50_view_100_rcrop_template_extra"
                     "rn50_view_100_valview_100_rcrop_template_default" "rn50_view_100_valview_100_rcrop_template_extra"
                     "vitb16_view_1_ccrop_template_default" "vitb16_view_1_ccrop_template_extra" 
                     "vitb16_view_100_rcrop_template_default" "vitb16_view_100_rcrop_template_extra"
                     "vitb16_view_100_valview_100_rcrop_template_default" "vitb16_view_100_valview_100_rcrop_template_extra")

for FEATURE in "${FEATURES[@]}"
do
    for DATASET in imagenet caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101
    do
        for SHOTS in 1 2 4 8 16
        do 
            for SEED in 1 2 3
            do
                python features.py \
                --dataset-config-file config/datasets/${DATASET}.yaml \
                --few-shot-config-file config/few_shot/shot_${SHOTS}.yaml \
                --features-config-file config/features/${FEATURE}.yaml \
                SEED ${SEED}
            done
        done
    done
done