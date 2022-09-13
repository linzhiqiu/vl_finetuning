#!/bin/bash

declare -a IMAGES=(
                   "rn50_layer_0"
                   "vitb16_layer_0" 
                   "rn50_layer_1"
                #    "vitb16_layer_1"
                #    "rn50_layer_2"
                #    "vitb16_layer_2"
                #    "vitb16_layer_4"
                #    "rn50_layer_all"
                #    "vitb16_layer_all"
                )

declare -a TEXTS=(
                  "layer_0" 
                #   "layer_1" 
                #   "layer_all"
                  )

declare -a TEMPLATES=(
                    #   "classname"
                    #   "default" 
                    #   "extra"
                      "single"
                     )

declare -a VIEWS=(
                  "view_1_ccrop"
                  "view_100_rcrop"
                  "view_100_valview_100_rcrop"
                 )
                  
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

declare -a CROSS_MODALS=(
                         "text_ratio_0"
                         "text_ratio_0.2"
                         "text_ratio_0.5"
                         "text_ratio_0.8"
                         "text_ratio_1"
                        )

declare -a LOGITS=(
                   "cosine_logit_scale"
                #    "cosine"
                   "linear"
                  )

declare -a HYPERS=(
                   "adamw"
                   "sgd"
                  )

declare -a ARCHITECTURES=(
                        #   "linear_bias"
                        #   "linear_zeroshot_bias"
                        #   "mlp_bias"
                          "linear"
                        #   "linear_zeroshot"
                        #   "mlp"
                         )
    
echo "IMAGES: ${IMAGES[@]}"
echo "TEXTS: ${TEXTS[@]}"
echo "TEMPLATES: ${TEMPLATES[@]}"
echo "VIEWS: ${VIEWS[@]}"
echo "DATASETS: ${DATASETS[@]}"
echo "CROSS_MODALS: ${CROSS_MODALS[@]}"
echo "LOGITS: ${LOGITS[@]}"
echo "HYPERS: ${HYPERS[@]}"
echo "ARCHITECTURES: ${ARCHITECTURES[@]}"

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
                    for SHOTS in 1 2 4 8 16
                    do 
                        echo "SHOTS: $SHOTS"
                        for SEED in 1 2 3
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
                                            SEED ${SEED}
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