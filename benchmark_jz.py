import pickle
import submitit

# synthcity absolute
from synthcity.benchmark import Benchmarks

import openml
# import GenericDataLoader
from synthcity.plugins.core.dataloader import GenericDataLoader
import os
from synthcity_addons import generators
from sklearn.preprocessing import QuantileTransformer
import hashlib

suite_id = 336
tasks = openml.study.get_suite(suite_id).tasks[:9]

tag = ["no_subsample_in_metric"]

gpu_models = [ "tabpfn_points", "ddpm", "ctgan", "tvae"]
cpu_models = ["arf", "smote", "forest_diffusion", "smote_imblearn", "gaussian_noise", "dummy_sampler"]
#cpu_models = ["dummy_sampler"]
#cpu_models = ["gaussian_noise"]

def run_model_on_dataset(model_name, task_id, hp_dic, save_results=True):
    from synthcity_addons import generators
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    dataset_name = dataset.name
    print(f"Running {model_name} on {dataset_name}")
    print("hp_dic", hp_dic)
    hp_str = "_".join([f"{k}_{v}" for k, v in hp_dic.items()]) # before it's modified by synthcity
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    # add target to X
    X["target"] = y
    # restrict X to 20K random samples
    if X.shape[0] > 20000:
        X = X.sample(20000, random_state=42)

    # restrict to 30 columns
    X = X.iloc[:, -30:]
    print(X.columns)
    normalization = hp_dic.pop("normalization", None)
    if normalization == "quantile":
        X = QuantileTransformer(n_quantiles=100, random_state=42).fit_transform(X)
    loader = GenericDataLoader(X, target_column="target")

    print("Loaded")


    score = Benchmarks.evaluate(
        [(model_name, model_name, hp_dic)],
        loader,
        synthetic_size=512,
        repeats=3,
        verbose=100,
        task_type="regression"
    )
    score_train = Benchmarks.evaluate(
        [(model_name, model_name, hp_dic)],
        loader,
        loader,
        synthetic_size=512,
        repeats=3,
        verbose=100,
        task_type="regression"
    )

    model_name = list(score.keys())[0]
    # change key names in score_train
    for key in score_train[model_name].keys():
        score_train[model_name][f"{key}_train"] = score_train[model_name].pop(key)
    #score[model_name].update(score_train[model_name])


    print("Benchmark done")
    if save_results:
        # save the results in results/dataset_id/model_name.pkl
        # create the folders if they don't exist
        os.makedirs("results_with_y", exist_ok=True)
        os.makedirs(f"results_with_y/{dataset_name}", exist_ok=True)
        print("hp_str", hp_str)
        with open(f"results_with_y/{dataset_name}/{model_name}_{hp_str}.pkl", "wb") as f:
                pickle.dump(score, f)
                with open(f"results_with_y/{dataset_name}/{model_name}_{hp_str}_train.pkl", "wb") as f:
                    pickle.dump(score_train, f)
    else:
        return score, score_train

def launch_jz_submission(config, gpu=True):
    config_copy = config.copy()
    print("config_copy", config_copy)
    #config_string = "_".join([f"{k}={v}" for k, v in config_copy.items()])
    # hash the config
    hash = hashlib.sha256(str(config_copy).encode()).hexdigest()
    with open(f"sbatch_files/sbatch_{hash}.sh", "w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=xval_{hash}\n")
        f.write(f"#SBATCH --output=xval_{hash}.out\n")
        f.write(f"#SBATCH --error=xval_{hash}.err\n")
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
        f.write(f"module load  pytorch-gpu/py3/1.13.0\n")
        #f.write(f"conda activate numerical_embeddings\n")
        command = "python evaluate_method_args.py"
        for key, value in config_copy.items():
            command += f" --{key} {value}"
        #f.write(f"python train_xval_args.py --{param} {variant} --iter {iter}\n")
        f.write(command)
    # submit the sbatch file
    print("Submitting sbatch file", f"sbatch_files/sbatch_{hash}.sh")
    os.system(f"sbatch sbatch_files/sbatch_{hash}.sh")
    print("Submitted")


for model in ["dummy_sampler", "gaussian_noise"]:
    for task in tasks:
        launch_jz_submission({"model": model, "task_id": task})
