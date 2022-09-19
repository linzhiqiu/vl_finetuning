#!/bin/bash

TOTAL=1
declare -a IMAGES=(
                   "rn50_layer_0"
                   "vitb16_layer_0" 
                #    "rn50_layer_1"
                #    "vitb16_layer_1"
                #    "rn50_layer_2"
                #    "vitb16_layer_2"
                #    "rn50_layer_all"
                #    "vitb16_layer_all"
                #    "vitb16_layer_4"
                )
TOTAL=$(( TOTAL * ${#IMAGES[@]} ))

declare -a TEXTS=(
                  "layer_0" 
                #   "layer_1" 
                #   "layer_all"
                  )
TOTAL=$(( TOTAL * ${#TEXTS[@]} ))

declare -a TEMPLATES=(
                      "classname"
                      "default" 
                      "extra"
                      "single"
                     )
TOTAL=$(( TOTAL * ${#TEMPLATES[@]} ))

declare -a VIEWS=(
                  "view_1_ccrop"
                  "view_10_valview_10_randomcrop"
                #   "view_100_rcrop"
                #   "view_100_valview_100_rcrop"
                 )
TOTAL=$(( TOTAL * ${#VIEWS[@]} ))

declare -a DATASETS=(
                     "imagenet"
                    #  "caltech101"
                    #  "dtd"
                    #  "eurosat"
                    #  "fgvc_aircraft"
                    #  "food101"
                    #  "oxford_flowers"
                    #  "oxford_pets"
                    #  "stanford_cars"
                    #  "sun397"
                    #  "ucf101" 
                     )
TOTAL=$(( TOTAL * ${#DATASETS[@]} ))

declare -a ALL_SHOTS=(
    "1"
    "2"
    "4"
    "8"
    "16"
)
TOTAL=$(( TOTAL * ${#ALL_SHOTS[@]} ))

declare -a ALL_SEEDS=(
    "1"
    "2"
    "3"
)
TOTAL=$(( TOTAL * ${#ALL_SEEDS[@]} ))

echo "IMAGES: ${IMAGES[@]}"
echo "TEXTS: ${TEXTS[@]}"
echo "TEMPLATES: ${TEMPLATES[@]}"
echo "VIEWS: ${VIEWS[@]}"
echo "DATASETS: ${DATASETS[@]}"
echo "ALL_SHOTS: ${ALL_SHOTS[@]}"
echo "ALL_SEEDS: ${ALL_SEEDS[@]}"
echo "TOTAL: $TOTAL"

COUNTER=1
echo " "
for DATASET in "${DATASETS[@]}"
do  
    echo "DATASET: $DATASET"
    for VIEW in "${VIEWS[@]}"
    do 
        echo "VIEW: $VIEW"
        for TEMPLATE in "${TEMPLATES[@]}"
        do
            echo "TEMPLATE: $TEMPLATE"
            for TEXT in "${TEXTS[@]}"
            do
                echo "TEXT: $TEXT"
                for IMAGE in "${IMAGES[@]}"
                do
                    echo "IMAGE: $IMAGE"
                    for SHOTS in "${ALL_SHOTS[@]}"
                    do 
                        echo "SHOTS: $SHOTS"
                        for SEED in "${ALL_SEEDS[@]}"
                        do
                            echo "SEED: $SEED"
                            echo "COUNTER: $COUNTER/$TOTAL"
                            echo " "
                            COUNTER=$(( COUNTER + 1 ))
                            python features.py \
                            --dataset-config-file config/datasets/${DATASET}.yaml \
                            --few-shot-config-file config/few_shot/shot_${SHOTS}.yaml \
                            --image-encoder-config-file config/features/image/${IMAGE}.yaml \
                            --text-encoder-config-file config/features/text/${TEXT}.yaml \
                            --template-config-file config/features/template/${TEMPLATE}.yaml \
                            --view-config-file config/features/view/${VIEW}.yaml \
                            SEED ${SEED}
                        done
                    done
                done
            done
        done
    done
done