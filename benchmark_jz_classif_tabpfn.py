import openml
import os
import hashlib

suite_id = 337
tasks = openml.study.get_suite(suite_id).tasks[:10]

tag = ["no_subsample_in_metric"]

gpu_models = [ "tabpfn_points_performance", "ddpm"]
cpu_models = ["smote", "gaussian_noise", "dummy_sampler", "oracle", "kmeans", "svm"]
#cpu_models = ["dummy_sampler"]

def launch_jz_submission(config, filename="", gpu=True):
    config_copy = config.copy()
    print("config_copy", config_copy)
    #config_string = "_".join([f"{k}={v}" for k, v in config_copy.items()])
    # hash the config
    #hash = hashlib.sha256(str(config_copy).encode()).hexdigest()
    # smaller hash
    hash = hashlib.sha256(str(config_copy).encode()).hexdigest()[:16]
    with open(f"sbatch_files/sbatch_{filename}_{hash}.sh", "w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=synth_{filename}_{hash}\n")
        f.write(f"#SBATCH --output=log_files/{filename}_{hash}.out\n")
        f.write(f"#SBATCH --error=log_files/{filename}_{hash}.err\n")
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
        command = "python evaluate_method_args_classif.py"
        for key, value in config_copy.items():
            command += f" --{key} {value}"
        #f.write(f"python train_xval_args.py --{param} {variant} --iter {iter}\n")
        f.write(command)
    # submit the sbatch file
    print("Submitting sbatch file", f"sbatch_files/sbatch_{filename}_{hash}.sh")
    os.system(f"sbatch sbatch_files/sbatch_{filename}_{hash}.sh")
    print("Submitted")


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
    "initialization_strategy": ["smote", "uniform"],
    "lr": [0.001, 0.01, 0.1],
    "n_batches": [100],
    #"n_batches": [50, 100, 200],
    "init_scale_factor": [5.],
    "n_permutations": [7],
    "n_ensembles": [7],
    "n_random_features_to_add": [1],
    #"random_test_points_scale": [2],
}

from itertools import product

for model in ["tabpfn_points_performance"]:#cpu_models + gpu_models: #["tabpfn_points"]:
    for task in tasks:
        for n_points in [10, 20, 30, 50, 100, 512]:
            for default_combination in product(*default_params.values()):
                config = dict(zip(default_params.keys(), default_combination))
                config["model_name"] = model
                config["task_id"] = task
                config["n_synthetic_points"] = n_points
                config["n_points_to_create"] = n_points
                launch_jz_submission(config, gpu=model in gpu_models, filename=f"{model}_{task}_{n_points}")
            #if model in ["kmeans", "tabpfn_points_performance"]:
            #launch_jz_submission({"model_name": model, "task_id": task, "n_synthetic_points": n_points,
            
            # elif model in ["svm"]:
            #     for C in [0.01, 0.1, 1., 10., 100.]:
            #         launch_jz_submission({"model_name": model, "task_id": task, "n_synthetic_points": n_points,
            #                               "C": C}, gpu=model in gpu_models, filename=f"{model}_{task}_{n_points}_{C}")
            # else:
            #     launch_jz_submission({"model_name": model, "task_id": task, "n_synthetic_points": n_points}, gpu=model in gpu_models, filename=f"{model}_{task}_{n_points}")
        # Grid search for default params
        # for default_combination in product(*default_params.values()):
        #     config = dict(zip(default_params.keys(), default_combination))
        #     config["model_name"] = model
        #     config["task_id"] = task
        #     launch_jz_submission(config, gpu=model in gpu_models, filename=f"{model}_{task}_default")
        #     #print(config)

            # # Iterate on the param variations (but not in grid)
            # for param_variation, variations in param_variations.items():
            #     for variation in variations:
            #         config_variation = config.copy()
            #         config_variation[param_variation] = variation
            #         print(config_variation)
            #        launch_jz_submission(config_variation, gpu=model in gpu_models, filename=f"{model}_{task}_{param_variation}_{variation}")
        # config = {}
        # config["model_name"] = model
        # config["task_id"] = task
        # #launch_jz_submission(config, gpu=True, filename=f"{model}_{task}_{param}_{variation}")
        # launch_jz_submission(config, gpu=model in gpu_models, filename=f"{model}_{task}")
                    # launch_jz_submission({"model_name": model, "task_id": task, "n_test_from_false_train": n_test_from_false_train, "n_batches": n_batches, "init_scale_factor": init_scale_factor}, gpu=True, 
                    #          filename=f"{model}_{task}")
