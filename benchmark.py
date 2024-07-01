import pickle
import submitit

# synthcity absolute
from synthcity.benchmark import Benchmarks

import openml
# import GenericDataLoader
from synthcity.plugins.core.dataloader import GenericDataLoader
import os
from synthcity_addons import generators

suite_id = 337
tasks = openml.study.get_suite(suite_id).tasks

gpu_models = ["ddpm", "ctgan", "tvae", "tabpfn_points"]
cpu_models = ["arf", "smote", "forest_diffusion"]
#gpu_models = []
#cpu_models = ["forest_diffusion"]

executor_gpu = submitit.AutoExecutor(folder="submitit_logs")
executor_gpu.update_parameters(timeout_min=2000, slurm_partition='parietal,normal,gpu-best,gpu', slurm_array_parallelism=7,#, cpus_per_task=2,
                                    gpus_per_node=1)
# executor.update_parameters(timeout_min=2000, slurm_partition='parietal,normal', slurm_array_parallelism=array_parallelism, cpus_per_task=2,
#                             exclude="margpu009")
executor_cpu = submitit.AutoExecutor(folder="submitit_logs")
executor_cpu.update_parameters(
                                        timeout_min=2000, slurm_partition='parietal,normal', slurm_array_parallelism=100,#, cpus_per_task=2,
                                        cpus_per_task=4, exclude="margpu009")


def run_model_on_dataset(model_name, task_id):
    from synthcity_addons import generators
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    dataset_name = dataset.name
    print(f"Running {model_name} on {dataset_name}")
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    # restrict X to 20K random samples
    if X.shape[0] > 20000:
        X = X.sample(20000, random_state=42)
    # restrict to 30 columns
    X = X.iloc[:, :30]
    loader = GenericDataLoader(X)

    score = Benchmarks.evaluate(
        [(model_name, model_name, {})],
        loader,
        synthetic_size=512,
        repeats=3,
    )

    # save the results in results/dataset_id/model_name.pkl
    # create the folders if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{dataset_name}", exist_ok=True)
    with open(f"results/{dataset_name}/{model_name}.pkl", "wb") as f:
        pickle.dump(score, f)

#run_model_on_dataset("forest_diffusion", tasks[1])

with executor_gpu.batch():
    for model in gpu_models:
        for task_id in tasks:
            executor_gpu.submit(run_model_on_dataset, model, task_id)

with executor_cpu.batch():
    for model in cpu_models:
        for task_id in tasks:
            executor_cpu.submit(run_model_on_dataset, model, task_id)
