#!/bin/bash

#SBATCH --job-name=NPH_shear_cuda
#SBATCH --output=/data1/shared/igraham/output/hoomd_prod/%A-%a.out
#SBATCH --time=1-00:00:00
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=0-100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titan_v:1
#SBATCH --mem=2G
#SBATCH --nodes=1

module load gcc/8.3.0
module load cuda/cuda-latest
module load shared/cuda/10.0
module load hpcx/2.5.0/hpcx

#conda activate softmatter

if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    # test case
    NUM=-1
else
    NUM=${SLURM_ARRAY_TASK_ID}
fi

# execute c++ code for all files with N=2048
# mkdir -p /data1/shared/igraham/datasets/fast_sim2
cd /home1/igraham/Projects/hoomd_test
/home1/igraham/anaconda3/envs/softmatter/bin/python NPH_shear.py -s ${NUM} -i 40 -m 1e-2 -n 10000 -p $(( 10**($NUM/) ))e-3
