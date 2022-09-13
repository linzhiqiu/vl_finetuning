#!/bin/bash

declare -a IMAGES=(
                   "rn50_layer_0"
                #    "vitb16_layer_0" 
                  )

declare -a TEXTS=(
                  "layer_0" 
                  )

declare -a TEMPLATES=(
                      "classname"
                      "default" 
                      "extra"
                      "single"
                     )

declare -a VIEWS=(
                  "view_1_ccrop"
                  "view_100_rcrop"
                  "view_100_valview_100_rcrop"
                 )
                  
declare -a DATASETS=(
                    #  "imagenet"
                    #  "caltech101"
                    #  "dtd"
                    #  "eurosat"
                    #  "fgvc_aircraft"
                    #  "food101"
                    #  "oxford_flowers"
                    #  "oxford_pets"
                    #  "stanford_cars"
                    #  "sun397"
                     "ucf101" 
                     )

declare -a CROSS_MODALS=(
                         "text_ratio_0"
                         "text_ratio_0.5"
                         "text_ratio_1"
                        )


    
echo "IMAGES: ${IMAGES[@]}"
echo "TEXTS: ${TEXTS[@]}"
echo "TEMPLATES: ${TEMPLATES[@]}"
echo "VIEWS: ${VIEWS[@]}"
echo "DATASETS: ${DATASETS[@]}"
echo "CROSS_MODALS: ${CROSS_MODALS[@]}"

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
                            for CROSS_MODAL in "${CROSS_MODALS[@]}"
                            do
                                echo "CROSS_MODAL: $CROSS_MODAL"
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
done