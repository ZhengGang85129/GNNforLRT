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

# Generate configs for inference
export sample_label="HSSPU200"
export model="TTBarPU200" # TTBarPU200, HNLPU200, MIXED

python3 make_inference.py \
    --sample_label $sample_label \
    --sample_hits_dir /global/cfs/cdirs/m3443/data/GNNforLRT/HSS_pgy_PU200 \
    --sample_particle_dir /global/cfs/cdirs/m3443/data/GNNforLRT/Hss_Pt1GeV_PU200_RAW/HSS_output_PU200 \
    --model $model \
    --test_split 1500

# Run inference
traintrack ./configs/pipeline_inference_$model-$sample_label.yaml --inference
