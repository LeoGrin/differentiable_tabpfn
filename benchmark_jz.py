import openml
import os
import hashlib

suite_id = 336
tasks = openml.study.get_suite(suite_id).tasks[:9]

tag = ["no_subsample_in_metric"]

gpu_models = [ "tabpfn_points", "ddpm", "ctgan", "tvae"]
cpu_models = ["arf", "smote", "forest_diffusion", "smote_imblearn", "gaussian_noise", "dummy_sampler"]
#cpu_models = ["dummy_sampler"]

def launch_jz_submission(config, filename="", gpu=True):
    config_copy = config.copy()
    print("config_copy", config_copy)
    #config_string = "_".join([f"{k}={v}" for k, v in config_copy.items()])
    # hash the config
    hash = hashlib.sha256(str(config_copy).encode()).hexdigest()
    with open(f"sbatch_files/sbatch_{hash}.sh", "w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=xval_{hash}\n")
        f.write(f"#SBATCH --output=log_files/{hash}.out\n")
        f.write(f"#SBATCH --error=log_files/{hash}.err\n")
        f.write(f"#SBATCH -n 1\n")
        f.write("#SBATCH --cpus-per-task=10\n")
        f.write("#SBATCH --ntasks-per-node=1\n")
        if gpu:
            f.write("#SBATCH --gpus-per-task=1\n")
        f.write("#SBATCH --hint=nomultithread\n")
        f.write("#SBATCH --time=20:00:00\n")
        if gpu:
            f.write("#SBATCH -A ptq@v100\n")
            #f.write("#SBATCH -C v100-32g\n")
        else:
            f.write("#SBATCH -A ptq@cpu\n")

##SBATCH --ntasks-per-node=${NUM_GPUS_PER_NODE}
#SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:${NUM_GPUS_PER_NODE}
#SBATCH --gpus-per-task=${NUM_GPUS_PER_NODE}
#SBATCH --hint=nomultithread         # hyperthreading desactive
        #f.write(f"module load pytorch-gpu/py3/2.1.1\n")
        f.write(f"module load pytorch-gpu/py3/1.13.0\n")
        #f.write(f"conda activate numerical_embeddings\n")
        command = "python evaluate_method_args.py"
        for key, value in config_copy.items():
            command += f" --{key} {value}"
        #f.write(f"python train_xval_args.py --{param} {variant} --iter {iter}\n")
        f.write(command)
    # submit the sbatch file
    print("Submitting sbatch file", f"sbatch_files/sbatch_{filename}_{hash}.sh")
    os.system(f"sbatch sbatch_files/sbatch_{filename}_{hash}.sh")
    print("Submitted")


for model in ["dummy_sampler", "gaussian_noise"]:
    for task in tasks:
        launch_jz_submission({"model_name": model, "task_id": task}, gpu=False, 
                             filename=f"{model}_{task}")
