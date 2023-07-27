#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH -n 4 -w cn47
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --open-mode=append
#SBATCH --error=../logs/model-%j.err
#SBATCH --output=../logs/model-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

project_dir=.

export WANDB_API_KEY=1bf134f08db03f667d7042b4500df8a0c7186d74
export WANDB_DIR=/ceph/csedu-scratch/other/jbrons/thesis-web-classification/wandb/
export WANDB_CACHE_DIR=/ceph/csedu-scratch/other/jbrons/thesis-web-classification/artifacts/

source "$project_dir"/venv/bin/activate
#python "$project_dir"/models/Newsgroup_SVM.py $SLURM_JOB_ID
python "$project_dir"/models/DMOZ_SVM.py $SLURM_JOB_ID
