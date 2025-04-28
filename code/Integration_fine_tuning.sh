#!/bin/bash
#SBATCH --job-name=Brain_metastsis    # Job name
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --time=3-00:00:00            # Max runtime: 10 hrs
#SBATCH --mem=500G                   # Memory request
#SBATCH --output=fine_tune_BM_2.log  # Output log file
#SBATCH --error=fine_tune_BM_2.log  # Error log file

# Activate your Conda environment
source ~/.bashrc
source /diazlab/data3/.abhinav/tools/miniconda3/etc/profile.d/conda.sh
source activate /diazlab/data3/.abhinav/tools/miniconda3/envs/py_r_env/
conda init
conda activate /diazlab/data3/.abhinav/tools/miniconda3/envs/py_r_env/

cd /diazlab/data3/.abhinav/projects/Brain_metastasis/
# Fine Tuning Integration
python /diazlab/data3/.abhinav/projects/Brain_metastasis/brain_metastasis_fine_tuning.py
