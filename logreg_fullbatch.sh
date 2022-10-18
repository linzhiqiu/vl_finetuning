#!/bin/bash
TOTAL=1
declare -a IMAGES=(
                   "rn50_layer_0"
                #    "vitb16_layer_0" 
                  )
TOTAL=$(( TOTAL * ${#IMAGES[@]} ))

declare -a TEXTS=(
                  "layer_0" 
                  )
TOTAL=$(( TOTAL * ${#TEXTS[@]} ))

declare -a TEMPLATES=(
                    #   "classname"
                    #   "default" 
                    #   "extra"
                      "single"
                     )
TOTAL=$(( TOTAL * ${#TEMPLATES[@]} ))

declare -a VIEWS=(
                  "view_1_ccrop"
                #   "view_100_rcrop"
                #   "view_10_valview_10_randomcrop"
                #   "view_100_valview_100_rcrop"
                 )
TOTAL=$(( TOTAL * ${#VIEWS[@]} ))

declare -a DATASETS=(
                    #  "imagenet"
                    #  "caltech101"
                    #  "dtd"
                    #  "eurosat"
                    #  "fgvc_aircraft"
                    #  "food101"
                     "oxford_flowers"
                     "oxford_pets"
                     "stanford_cars"
                     "sun397"
                     "ucf101"
                     )
TOTAL=$(( TOTAL * ${#DATASETS[@]} ))

declare -a CROSS_MODALS=(
                        #  "text_ratio_1"
                        #  "text_ratio_0.5"
                        #  "normtext_ratio_0.5"
                         "normboth_ratio_0.5"
                        #  "text_ratio_0"
                        )
TOTAL=$(( TOTAL * ${#CROSS_MODALS[@]} ))

declare -a ALL_SHOTS=(
    # "1"
    # "2"
    # "4"
    # "8"
    "16"
)
TOTAL=$(( TOTAL * ${#ALL_SHOTS[@]} ))


    
echo "IMAGES: ${IMAGES[@]}"
echo "TEXTS: ${TEXTS[@]}"
echo "TEMPLATES: ${TEMPLATES[@]}"
echo "VIEWS: ${VIEWS[@]}"
echo "DATASETS: ${DATASETS[@]}"
echo "CROSS_MODALS: ${CROSS_MODALS[@]}"
echo "ALL_SHOTS: ${ALL_SHOTS[@]}"
echo "TOTAL: $TOTAL"

COUNTER=0


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
                        for CROSS_MODAL in "${CROSS_MODALS[@]}"
                        do
                            echo "CROSS_MODAL: $CROSS_MODAL"
                            (( COUNTER++ ))
                            echo "COUNTER: $COUNTER/$TOTAL"
                            python logreg_fullbatch.py \
                            --dataset-config-file config/datasets/${DATASET}.yaml \
                            --few-shot-config-file config/few_shot/shot_${SHOTS}.yaml \
                            --image-encoder-config-file config/features/image/${IMAGE}.yaml \
                            --text-encoder-config-file config/features/text/${TEXT}.yaml \
                            --template-config-file config/features/template/${TEMPLATE}.yaml \
                            --view-config-file config/features/view/${VIEW}.yaml \
                            --cross-modal-config-file config/cross_modal/${CROSS_MODAL}.yaml \
                            --hyperparams-config-file config/hyperparams/logreg_fullbatch/default.yaml
                        done
                    done
                done
            done
        done
    done
done