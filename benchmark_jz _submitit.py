import openml
import os
import hashlib
import submitit
from itertools import product

suite_id = 336
tasks = openml.study.get_suite(suite_id).tasks[:10]

tag = ["no_subsample_in_metric"]

gpu_models = ["tabpfn_points", "ddpm", "ctgan", "tvae", "tabpfn_generator"]
cpu_models = ["arf", "smote", "forest_diffusion", "smote_imblearn", "gaussian_noise", "dummy_sampler", "oracle"]
#cpu_models = ["dummy_sampler"]

def run_evaluation(config):
    command = "python evaluate_method_args.py"
    for key, value in config.items():
        command += f" --{key} {value}"
    os.system(command)

param_variations = {
    "n_test_from_false_train": [256], 
    "n_batches": [10, 20, 250], 
    "loss": ["average"],
    #"init_scale_factor": [10.],
    #"n_permutations": [5, 7],
    #"n_ensembles": [5, 7],
    #"n_random_features_to_add": [0, 2, 5],
    "random_test_points_scale": [3],
}

default_params = {
    #"initialization_strategy": ["gaussian_noise", "smote", "uniform"],
    "lr": [1e-4, 1e-3],
    "model": ["mlp", "resnet"],
    "n_layers": [3, 6],
    "d_hidden": [256, 512],
    #"loss": ["individual"],
    #"n_test_from_false_train": [0],
    "n_batches": [150],
    "init_scale_factor": [5.],
    "n_permutations": [5],
    "n_ensembles": [5],
    "n_random_features_to_add": [0, 1],
    "random_test_points_scale": [2],
}

# Create the executor once
executor_gpu = submitit.AutoExecutor(folder="submitit_logs")
executor_gpu.update_parameters(timeout_min=2000, slurm_partition='parietal,normal,gpu-best,gpu', slurm_array_parallelism=5,#, cpus_per_task=2,
                                    gpus_per_node=1)
# executor.update_parameters(timeout_min=2000, slurm_partition='parietal,normal', slurm_array_parallelism=array_parallelism, cpus_per_task=2,
#                             exclude="margpu009")
executor_cpu = submitit.AutoExecutor(folder="submitit_logs")
executor_cpu.update_parameters(
                                        timeout_min=2000, slurm_partition='parietal,normal', slurm_array_parallelism=100,#, cpus_per_task=2,
                                        cpus_per_task=4, exclude="margpu009")
jobs = []

for model in ["tabpfn_generator"]:  # cpu_models + gpu_models: #["tabpfn_points"]:
    executor = executor_gpu if model in gpu_models else executor_cpu
    executor.update_parameters(job_name=f"{model}_benchmark")
    with executor.batch():  
        for task in tasks:
            # Grid search for default params
            for default_combination in product(*default_params.values()):
                config = dict(zip(default_params.keys(), default_combination))
                config["model_name"] = model
                config["task_id"] = task
                executor.submit(run_evaluation, config)
