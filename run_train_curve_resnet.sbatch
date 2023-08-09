#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --qos=regular
#SBATCH --account=m636_g
#SBATCH -o logs/%N.%j..out # STDOUT
#SBATCH -e logs/%N.%j..err # STDERR


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
export PYTHONUNBUFFERED=1



###############################################################################
### run scripts here
###############################################################################

### resnet20_batch_norm_True_residual_True

python3 train_resnet.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_True_seed_0_seed_123 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_True --epochs=50 --lr=1e-3 --wd=5e-4 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_True_seed_0_net.pkl --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_True_seed_123_net.pkl --fix_start --fix_end

python3 eval_curve.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_True_seed_0_seed_123 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_True --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_True_seed_0_seed_123/checkpoint-50.pt --num_points=5



python3 train_resnet.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_True_seed_0_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_True --epochs=50 --lr=1e-3 --wd=5e-4 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_True_seed_0_net.pkl --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_True_seed_2023_net.pkl --fix_start --fix_end

python3 eval_curve.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_True_seed_0_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_True --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_True_seed_0_seed_2023/checkpoint-50.pt --num_points=5



python3 train_resnet.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_True_seed_123_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_True --epochs=50 --lr=1e-3 --wd=5e-4 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_True_seed_123_net.pkl --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_True_seed_2023_net.pkl --fix_start --fix_end

python3 eval_curve.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_True_seed_123_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_True --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_True_seed_123_seed_2023/checkpoint-50.pt --num_points=5



### resnet20_batch_norm_True_residual_False

python3 train_resnet.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_False_seed_0_seed_123 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_False --epochs=50 --lr=1e-3 --wd=5e-4 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_False_seed_0_net.pkl --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_False_seed_123_net.pkl --fix_start --fix_end

python3 eval_curve.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_False_seed_0_seed_123 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_False --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_False_seed_0_seed_123/checkpoint-50.pt --num_points=5



python3 train_resnet.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_False_seed_0_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_False --epochs=50 --lr=1e-3 --wd=5e-4 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_False_seed_0_net.pkl --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_False_seed_2023_net.pkl --fix_start --fix_end

python3 eval_curve.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_False_seed_0_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_False --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_False_seed_0_seed_2023/checkpoint-50.pt --num_points=5



python3 train_resnet.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_False_seed_123_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_False --epochs=50 --lr=1e-3 --wd=5e-4 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_False_seed_123_net.pkl --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/ResNet20_loss_lens/resnet20_batch_norm_True_residual_False_seed_2023_net.pkl --fix_start --fix_end

python3 eval_curve.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_False_seed_123_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=resnet20_batch_norm_True_residual_False --wd=5e-4 --curve=Bezier --num_bends=3 --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/resnet20_batch_norm_True_residual_False_seed_123_seed_2023/checkpoint-50.pt --num_points=5






###############################################################################
### finish
###############################################################################

wait
date