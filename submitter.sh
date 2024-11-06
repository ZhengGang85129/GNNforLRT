#!/usr/bin/bash
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=2
#SBATCH -c 32
#SBATCH -Am3443
#SBATCH --time 24:0:0
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=disney85129@gmail.com
#SBATCH -o ./sinfo/output.%J.out
export SLURM_CPU_BIND="cores"
cd /global/homes/z/zhenggan/workspace/Project
pwd
bash
nvidia-smi
which bash
source /global/homes/z/zhenggan/miniconda3/etc/profile.d/conda.sh
conda activate exatrkx-gpu
traintrack configs/pipeline-mixing_PU200_smeared.yaml #> output-PU200_GNN-0099.log 2>&1 
mv stage_performance.png stage_performance-PU200-smeared_all.png
mv stage_performance.pdf stage_performance-PU200-smeared_all.pdf
