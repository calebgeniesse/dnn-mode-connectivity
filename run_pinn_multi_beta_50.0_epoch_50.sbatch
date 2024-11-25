#!/bin/bash

#SBATCH --job-name=run_pinn_multi_beta_50.0_epoch_50
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --qos=shared
#SBATCH --account=m636
#SBATCH -o logs/run_pinn_multi_beta_50.0_epoch_50.%j.out # STDOUT
#SBATCH -e logs/run_pinn_multi_beta_50.0_epoch_50.%j.err # STDERR


###############################################################################
### setup here
###############################################################################

pwd
hostname
date
echo starting job...
source ~/.bashrc
module load python
# conda activate py39
# module load pytorch/2.0.1
mamba activate /global/cfs/cdirs/m636/geniesse/envs/pinns
export PYTHONUNBUFFERED=1



###############################################################################
### run scripts here
###############################################################################

    
python3 train_eval_pinn_multi.py \
    --dir=/global/cfs/cdirs/m636/geniesse/projects/dnn-mode-connectivity/checkpoints_global \
    --dataset=CIFAR10 \
    --data_path=/global/cfs/cdirs/m636/geniesse/data \
    --transform=ToTensor \
    --model=PINNDNN \
    --curve=Bezier \
    --num_bends=3 \
    --fix_start --fix_end \
    --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 \
    --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=50 --u0_str='sin(x)' --source=0 \
    --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' \
    --init_file=/global/cfs/cdirs/m636/geniesse/projects/dnn-mode-connectivity/PINN_convection_beta_50.0_lr_1.0_n_seeds_100_pairs.csv \
    --input_folder /global/cfs/cdirs/m636/geniesse/projects/characterizing-pinns-failure-modes/pbc_examples/checkpoints \
    --output_folder /global/cfs/cdirs/m636/geniesse/projects/dnn-mode-connectivity/checkpoints_global \
    --epochs=50 \
    --eval_epoch 50 \
    --num_points=5 



# --use_test
# --linear-init



###############################################################################
### finish
###############################################################################

wait
date