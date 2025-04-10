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
#SBATCH --mail-user=a0910555246@gmail.com
export SLURM_CPU_BIND="cores"
cd /global/homes/z/zhenggan/workspace/Project
pwd
bash
nvidia-smi
which bash
source /global/homes/z/zhenggan/miniconda3/etc/profile.d/conda.sh
conda activate exatrkx-gpu
traintrack configs/inference_test/pipeline-BIN00.yaml --inference > output-FlatPU.log 2>&1 


