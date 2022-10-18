#!/bin/bash

WORKERS=0

TOTAL=1
declare -a IMAGES=(
                   "rn50_layer_0"
                #    "vitb16_layer_0" 
                )
TOTAL=$(( TOTAL * ${#IMAGES[@]} ))

declare -a TEXTS=(
                  "layer_0"
                #   "layer_1" 
                #   "layer_all"
                  )
TOTAL=$(( TOTAL * ${#TEXTS[@]} ))

declare -a TEMPLATES=(
                      "default"
                     )
TOTAL=$(( TOTAL * ${#TEMPLATES[@]} ))

declare -a VIEWS=(
                  "view_1_ccrop"
                 )
TOTAL=$(( TOTAL * ${#VIEWS[@]} ))
                  
declare -a DATASETS=(
                     "dtd"
                     "eurosat"
                     "fgvc_aircraft"
                     "food101"
                     "oxford_flowers"
                     "oxford_pets"
                     "ucf101" 
                     )
TOTAL=$(( TOTAL * ${#DATASETS[@]} ))

declare -a CROSS_MODALS=(
                         "normtext_ratio_0.5"
                         "normtext_ratio_0.2"
                         "normtext_ratio_0.8"
                        )
TOTAL=$(( TOTAL * ${#CROSS_MODALS[@]} ))

declare -a LOGITS=(
                #    "cosine_logit_scale"
                #    "cosine"
                   "linear"
                  )
TOTAL=$(( TOTAL * ${#LOGITS[@]} ))

declare -a HYPERS=(
                   "adamw_2"
                  )
TOTAL=$(( TOTAL * ${#HYPERS[@]} ))

declare -a ARCHITECTURES=(
                        #   "linear_bias"
                        #   "linear_zeroshot_bias"
                        #   "mlp_bias"
                          "linear"
                        #   "linear_zeroshot"
                        #   "mlp"
                         )
TOTAL=$(( TOTAL * ${#ARCHITECTURES[@]} ))

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
echo "CROSS_MODALS: ${CROSS_MODALS[@]}"
echo "LOGITS: ${LOGITS[@]}"
echo "HYPERS: ${HYPERS[@]}"
echo "ARCHITECTURES: ${ARCHITECTURES[@]}"
echo "ALL_SHOTS: ${ALL_SHOTS[@]}"
echo "ALL_SEEDS: ${ALL_SEEDS[@]}"
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
                        for SEED in "${ALL_SEEDS[@]}"
                        do
                            echo "SEED: $SEED"
                            for CROSS_MODAL in "${CROSS_MODALS[@]}"
                            do
                                echo "CROSS_MODAL: $CROSS_MODAL"
                                for LOGIT in "${LOGITS[@]}"
                                do
                                    echo "LOGIT: $LOGIT"
                                    for HYP in "${HYPERS[@]}"
                                    do
                                        echo "HYP: $HYP"
                                        for ARCH in "${ARCHITECTURES[@]}"
                                        do
                                            echo "ARCH: $ARCH"
                                            (( COUNTER++ ))
                                            echo "COUNTER: $COUNTER/$TOTAL"
                                            python logreg_minibatch.py \
                                            --dataset-config-file config/datasets/${DATASET}.yaml \
                                            --few-shot-config-file config/few_shot/shot_${SHOTS}.yaml \
                                            --image-encoder-config-file config/features/image/${IMAGE}.yaml \
                                            --text-encoder-config-file config/features/text/${TEXT}.yaml \
                                            --template-config-file config/features/template/${TEMPLATE}.yaml \
                                            --view-config-file config/features/view/${VIEW}.yaml \
                                            --cross-modal-config-file config/cross_modal/${CROSS_MODAL}.yaml \
                                            --logit-config-file config/logit/${LOGIT}.yaml \
                                            --hyperparams-config-file config/hyperparams/logreg_minibatch/${HYP}.yaml \
                                            --architecture-config-file config/architecture/${ARCH}.yaml \
                                            SEED ${SEED} \
                                            DATALOADER.NUM_WORKERS ${WORKERS}
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done