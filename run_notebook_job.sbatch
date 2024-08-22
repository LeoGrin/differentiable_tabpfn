#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=parietal,normal,gpu-best,gpu
#SBATCH --exclude=margpu002,margpu003,margpu004
#SBATCH --gpus=1
# Activate the specified environment
micromamba activate "dpo"

# Launch Jupyter Notebook without opening a browser and on the specified port
jupyter-notebook --no-browser --port=8000 --NotebookApp.token=8fc62f364f38046445398152 --NotebookApp.allow_origin_pat=https://.*vscode-cdn\.net
