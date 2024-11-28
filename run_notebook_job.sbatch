#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=parietal,normal,gpu-best,gpu
##SBATCH --exclude=margpu002,margpu003,margpu004
#SBATCH --gpus=1
# Activate the specified environment
micromamba activate "dpo"

# Launch Jupyter Notebook without opening a browser and on the specified port
jupyter-notebook --no-browser --port=8888 --NotebookApp.token=a3cfdf20995aa917051eb41d --NotebookApp.allow_origin_pat=https://.*vscode-cdn\.net
