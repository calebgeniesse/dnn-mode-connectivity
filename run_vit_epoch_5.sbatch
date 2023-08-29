#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --qos=regular
#SBATCH --account=m636_g
#SBATCH -o logs/vit.%N.%j..out # STDOUT
#SBATCH -e logs/vit.%N.%j..err # STDERR


###############################################################################
### setup here
###############################################################################

pwd
hostname
date
echo starting job...
source ~/.bashrc
module load python
conda activate py39
module load pytorch/2.0.1
bash install_robustbench.sh

export PYTHONUNBUFFERED=1



###############################################################################
### run scripts here
###############################################################################
### VIT_model_seed_SEED_threshold_\{00\}

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_0_seed_123 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_0_threshold_\{00\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123_threshold_\{00\}.pt --fix_start --fix_end --threshold=00 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_0_seed_123/checkpoint-5.pt --num_points=5

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_0_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_0_threshold_\{00\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_2023_threshold_\{00\}.pt --fix_start --fix_end --threshold=00 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_0_seed_2023/checkpoint-5.pt --num_points=5

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_0_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_0_threshold_\{00\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123456_threshold_\{00\}.pt --fix_start --fix_end --threshold=00 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_0_seed_123456/checkpoint-5.pt --num_points=5

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_123_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123_threshold_\{00\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_2023_threshold_\{00\}.pt --fix_start --fix_end --threshold=00 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_123_seed_2023/checkpoint-5.pt --num_points=5

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_123_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123_threshold_\{00\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123456_threshold_\{00\}.pt --fix_start --fix_end --threshold=00 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_123_seed_123456/checkpoint-5.pt --num_points=5

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_2023_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_2023_threshold_\{00\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123456_threshold_\{00\}.pt --fix_start --fix_end --threshold=00 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{00\}_seed_2023_seed_123456/checkpoint-5.pt --num_points=5


### VIT_model_seed_SEED_threshold_\{100\}

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_0_seed_123 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_0_threshold_\{100\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123_threshold_\{100\}.pt --fix_start --fix_end --threshold=100 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_0_seed_123/checkpoint-5.pt --num_points=5

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_0_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_0_threshold_\{100\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_2023_threshold_\{100\}.pt --fix_start --fix_end --threshold=100 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_0_seed_2023/checkpoint-5.pt --num_points=5

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_0_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_0_threshold_\{100\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123456_threshold_\{100\}.pt --fix_start --fix_end --threshold=100 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_0_seed_123456/checkpoint-5.pt --num_points=5

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_123_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123_threshold_\{100\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_2023_threshold_\{100\}.pt --fix_start --fix_end --threshold=100 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_123_seed_2023/checkpoint-5.pt --num_points=5

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_123_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123_threshold_\{100\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123456_threshold_\{100\}.pt --fix_start --fix_end --threshold=100 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_123_seed_123456/checkpoint-5.pt --num_points=5

python3 train_eval_vit.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_2023_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=ViT --epochs=5 --lr=1e-4 --wd=5e-5 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_2023_threshold_\{100\}.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/Vit_loss_lens/VIT_model_seed_123456_threshold_\{100\}.pt --fix_start --fix_end --threshold=100 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/VIT_model_seed_SEED_threshold_\{100\}_seed_2023_seed_123456/checkpoint-5.pt --num_points=5

###############################################################################
### finish
###############################################################################

wait
date