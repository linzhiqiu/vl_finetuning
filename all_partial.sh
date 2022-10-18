#!/bin/bash

WORKERS=0

TOTAL=1
declare -a IMAGES=(
                   "rn50_layer_1"
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
                    #   "extra_default"
                      "single"
                     )
TOTAL=$(( TOTAL * ${#TEMPLATES[@]} ))

declare -a VIEWS=(
                #   "view_10_valview_10_randomcrop" # not good
                  "view_1_ccrop"
                #   "view_100_valview_100_rcrop" # don't run this now
                 )
TOTAL=$(( TOTAL * ${#VIEWS[@]} ))
                  
declare -a DATASETS=(
                     "imagenet" # running for 1 and 16 shot
                    #  "food101"
                    #  "caltech101"
                    #  "dtd"
                    #  "eurosat"
                    #  "fgvc_aircraft"
                    #  "oxford_flowers"
                    #  "oxford_pets"
                    #  "stanford_cars"
                    #  "sun397"
                    #  "ucf101" 
                     )
TOTAL=$(( TOTAL * ${#DATASETS[@]} ))

declare -a CROSS_MODALS=(
                        #  "text_ratio_1"
                        #  "text_ratio_0.2"
                        #  "text_ratio_0.8"
                        #  "text_ratio_0.5"
                         "text_ratio_0"
                         "normtext_ratio_0.5"
                        )
TOTAL=$(( TOTAL * ${#CROSS_MODALS[@]} ))

declare -a LOGITS=(
    # "fnorm_False_hnorm_False_logit_Fixed"
    # "fnorm_False_hnorm_True_logit_Fixed"
    "fnorm_True_hnorm_False_logit_Fixed"
    # "fnorm_True_hnorm_True_logit_Fixed"
    "fnorm_False_hnorm_False_logit_Learn"
    # "fnorm_False_hnorm_True_logit_Learn"
    # "fnorm_True_hnorm_False_logit_Learn"
    # "fnorm_True_hnorm_True_logit_Learn"
                  )
TOTAL=$(( TOTAL * ${#LOGITS[@]} ))

declare -a HYPERS=(
                   "partial_adamw_fast"
                  )
TOTAL=$(( TOTAL * ${#HYPERS[@]} ))

declare -a ARCHITECTURES=(
                        #   "linear_bias"
                        #   "linear_zeroshot_bias"
                        #   "mlp_bias"
                        #   "linear"
                          "linear_zeroshot"
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
    # "3"
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
                                            python final_logit_ablation.py \
                                            --dataset-config-file config/datasets/${DATASET}.yaml \
                                            --few-shot-config-file config/few_shot/shot_${SHOTS}.yaml \
                                            --image-encoder-config-file config/features/image/${IMAGE}.yaml \
                                            --text-encoder-config-file config/features/text/${TEXT}.yaml \
                                            --template-config-file config/features/template/${TEMPLATE}.yaml \
                                            --view-config-file config/features/view/${VIEW}.yaml \
                                            --cross-modal-config-file config/cross_modal/${CROSS_MODAL}.yaml \
                                            --logit-config-file config/final_logit/${LOGIT}.yaml \
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

echo "Start testing..."
python eval_all_partial.py