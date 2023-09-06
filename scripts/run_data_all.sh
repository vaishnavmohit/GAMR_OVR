#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
##SBATCH -C v100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohit_vaishnav@brown.edu
#SBATCH -n 4
#SBATCH --mem=30G
#SBATCH -N 1

# Specify a job name:
#SBATCH --time=40:00:00

# Specify an output file
#SBATCH -o ../../slurm/OVR/%j.out
#SBATCH -e ../../slurm/OVR/%j.err

#SBATCH -C quadrortx 
#SBATCH --account=carney-tserre-condo

# activate conda env
module load cuda/10.2 # 11.1.1 # 
# module load gcc/8.3
source ../../env/visreason/bin/activate
module load python/3.7.4

# debugging flags (optional)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTHONFAULTHANDLER=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
echo $CUDA_VISIBLE_DEVICES
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODgwMGZmNjktNWMyYS00NjViLWE2MjAtNjY5YWQ1ZmUzNGFmIn0="
export HYDRA_FULL_ERROR=1

r=$((1 + $RANDOM % 10))

echo $r

name=gamr
gpu=1
b_s=64
steps=${5:-4}

data_type=${2:-AB} # AB or SD
key=${1:-Base} # Base or NonTrivial
lr=${3:-.0001}
w_d=${4:-.0001}

sleep ${r}m

python train_v3.py --config-path=config --config-name=config model.architecture=${name} \
                    trainer.gpus=${gpu} trainer.max_epochs=100 training.data_type=${data_type} \
                    training.neptune=False training.optuna=False training.key=${key} \
                    training.nclasses=2 training.batch_size=${b_s} \
                    training.num_workers=3 model.steps=${steps} model_optuna.lr=${lr} \
                    model_optuna.weight_decay=${w_d}
