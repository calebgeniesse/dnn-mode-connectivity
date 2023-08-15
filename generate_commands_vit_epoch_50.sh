#!/bin/bash

THRESHOLDS=(
    "00"
    "100"
)
# MODELS=(
#     "VIT_model_seed_SEED_threshold_{00}"
#     "VIT_model_seed_SEED_threshold_{100}"
# )

# TODO: can we skip training if we use linear initialization?
EPOCHS=50
EVAL_EPOCH=50


for THRESHOLD in "${THRESHOLDS[@]}" ; do

    MODEL_1="VIT_model_seed_SEED_threshold_\{${THRESHOLD}\}"

    echo ""
    echo "###" ${MODEL_1}
    echo ""
    
    # MODEL_1="resnet20_batch_norm_True_residual_True"
    MODE_PATH="/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/${MODEL_1}.pt"
    SAVE_PATH="/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints"


    SEEDS=(0 123 2023 123456)
    # SEEDS=(0 123 2023)

    for SEED_1 in "${SEEDS[@]}" ; do
        for SEED_2 in "${SEEDS[@]}" ; do

            if [[ "${SEED_1}" == "${SEED_2}" ]] ; then
                continue;
            fi
            if [[ "${SEED_1}" -gt "${SEED_2}" ]] ; then
                continue;
            fi

            MODE_1="${MODE_PATH/SEED/${SEED_1}}"
            MODE_2="${MODE_PATH/SEED/${SEED_2}}"

            CHECKPOINTS_DIR=${SAVE_PATH}/${MODEL_1}_seed_${SEED_1}_seed_${SEED_2}
            mkdir -p ${CHECKPOINTS_DIR}
            
            CHECKPOINT_1=${CHECKPOINTS_DIR}/checkpoint-${EVAL_EPOCH}.pt

            # THRESHOLD=0
            
            # echo ${MODE_1}
            # echo ${MODE_2}
            # echo ""

            _train_cmd="python3 train_eval_vit.py \
                    --dir=${CHECKPOINTS_DIR} \
                    --dataset=CIFAR10 \
                    --data_path=/global/homes/g/geniesse/data \
                    --transform=ToTensor \
                    --model=ViT \
                    --epochs=${EPOCHS} \
                    --lr=1e-3 \
                    --wd=5e-4 \
                    --curve=Bezier \
                    --num_bends=3 \
                    --init_start=${MODE_1} \
                    --init_end=${MODE_2} \
                    --fix_start \
                    --fix_end \
                    --threshold=${THRESHOLD} \
                    --ckpt=${CHECKPOINT_1} \
                    --num_points=5"
                    #
                    # --use_test"
                    # --linear-init
                    
             
                    

            # echo ""
            echo $_train_cmd
            # echo ""
            echo ""

        done
    done
    
    # echo ""
    
done
		 
