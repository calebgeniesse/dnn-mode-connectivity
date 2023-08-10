#!/bin/bash

MODELS=(
    "resnet20_batch_norm_True_residual_True"
    "resnet20_batch_norm_True_residual_False"
)

# TODO: can we skip training if we use linear initialization?
EPOCHS=1
EVAL_EPOCH=0


for MODEL_1 in "${MODELS[@]}" ; do

    echo ""
    echo "###" ${MODEL_1}
    echo ""
    
    # MODEL_1="resnet20_batch_norm_True_residual_True"
    MODE_PATH="/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/${MODEL_1}_seed_SEED_net.pkl"
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

            # echo ${MODE_1}
            # echo ${MODE_2}
            # echo ""

            _train_cmd="python3 train_resnet.py \
                    --dir=${CHECKPOINTS_DIR} \
                    --dataset=CIFAR10 \
                    --data_path=/global/homes/g/geniesse/data \
                    --transform=ToTensor \
                    --model=${MODEL_1} \
                    --epochs=${EPOCHS} \
                    --lr=1e-3 \
                    --wd=5e-4 \
                    --curve=Bezier \
                    --num_bends=3 \
                    --init_start=${MODE_1} \
                    --init_end=${MODE_2} \
                    --fix_start \
                    --fix_end"
                    #
                    # --use_test"
                    # --linear-init
                    
             _eval_cmd="python3 eval_curve.py \
                    --dir=${CHECKPOINTS_DIR} \
                    --dataset=CIFAR10 \
                    --data_path=/global/homes/g/geniesse/data \
                    --transform=ToTensor \
                    --model=${MODEL_1} \
                    --wd=5e-4 \
                    --curve=Bezier \
                    --num_bends=3 \
                    --ckpt=${CHECKPOINT_1} \
                    --num_points=5"
                    
                    # --use_test"
                    # --linear-init
                    

            # echo ""
            echo $_train_cmd
            echo ""
            echo $_eval_cmd
            echo ""
            echo ""

        done
    done
    
    # echo ""
    
done
		 
