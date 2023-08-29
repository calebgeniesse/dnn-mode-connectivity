#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --qos=regular
#SBATCH --account=m636_g
#SBATCH -o logs/pinn.%N.%j..out # STDOUT
#SBATCH -e logs/pinn.%N.%j..err # STDERR


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

### PINN_convection_beta_1

python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_0_seed_123 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_0.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_123.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=1 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_0_seed_123/checkpoint-0.pt --num_points=5




python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_0_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_0.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_2023.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=1 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_0_seed_2023/checkpoint-0.pt --num_points=5




python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_0_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_0.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_123456.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=1 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_0_seed_123456/checkpoint-0.pt --num_points=5




python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_123_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_123.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_2023.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=1 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_123_seed_2023/checkpoint-0.pt --num_points=5




python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_123_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_123.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_123456.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=1 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_123_seed_123456/checkpoint-0.pt --num_points=5




python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_2023_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_2023.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_1_seed_123456.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=1 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_1_seed_2023_seed_123456/checkpoint-0.pt --num_points=5





### PINN_convection_beta_50

python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_0_seed_123 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_0.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_123.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=50 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_0_seed_123/checkpoint-0.pt --num_points=5




python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_0_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_0.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_2023.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=50 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_0_seed_2023/checkpoint-0.pt --num_points=5




python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_0_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_0.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_123456.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=50 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_0_seed_123456/checkpoint-0.pt --num_points=5




python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_123_seed_2023 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_123.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_2023.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=50 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_123_seed_2023/checkpoint-0.pt --num_points=5




python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_123_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_123.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_123456.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=50 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_123_seed_123456/checkpoint-0.pt --num_points=5




python3 train_eval_pinn.py --dir=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_2023_seed_123456 --dataset=CIFAR10 --data_path=/global/homes/g/geniesse/data --transform=ToTensor --model=PINNDNN --epochs=1 --curve=Bezier --num_bends=3 --init_start=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_2023.pt --init_end=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_loss_lens/PINN_convection_beta_50_seed_123456.pt --fix_start --fix_end --system='convection' --seed=0 --N_f=100 --optimizer_name='LBFGS' --lr=1e-4 --wd=5e-5 --L=1.0 --xgrid=256 --nt=100 --nu=1.0 --rho=1.0 --beta=50 --u0_str='sin(x)' --source=0 --layers='50,50,50,50,1' --net='DNN' --activation='tanh' --loss_style='mean' --ckpt=/global/homes/g/geniesse/dnn-mode-connectivity/checkpoints/PINN_convection_beta_50_seed_2023_seed_123456/checkpoint-0.pt --num_points=5



###############################################################################
### finish
###############################################################################

wait
date