#!/usr/bin/bash
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -Am3443
#SBATCH --time 24:0:0
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=daniel.bb0321@gmail.com
export SLURM_CPU_BIND="cores"
module load conda 
conda activate trackml

cd ~/GNNforLRT

# Run inference
traintrack ./configs/pipeline-HNL_best.yaml
