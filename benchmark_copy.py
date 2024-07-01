import pickle
#import submitit

# synthcity absolute
from synthcity.benchmark import Benchmarks

import openml
# import GenericDataLoader
from synthcity.plugins.core.dataloader import GenericDataLoader
import os
from synthcity_addons import generators
from sklearn.preprocessing import QuantileTransformer

suite_id = 336
tasks = openml.study.get_suite(suite_id).tasks[:9]

tag = ["no_subsample_in_metric"]

gpu_models = [ "tabpfn_points", "ddpm", "ctgan", "tvae"]
cpu_models = ["arf", "smote", "forest_diffusion", "smote_imblearn", "gaussian_noise", "dummy_sampler"]
#cpu_models = ["dummy_sampler"]
#cpu_models = ["gaussian_noise"]


# executor_gpu = submitit.AutoExecutor(folder="submitit_logs")
# executor_gpu.update_parameters(timeout_min=2000, slurm_partition='parietal,normal,gpu-best,gpu', slurm_array_parallelism=5,#, cpus_per_task=2,
#                                     gpus_per_node=1)
# # executor.update_parameters(timeout_min=2000, slurm_partition='parietal,normal', slurm_array_parallelism=array_parallelism, cpus_per_task=2,
# #                             exclude="margpu009")
# executor_cpu = submitit.AutoExecutor(folder="submitit_logs")
# executor_cpu.update_parameters(
#                                         timeout_min=2000, slurm_partition='parietal,normal', slurm_array_parallelism=100,#, cpus_per_task=2,
#                                         cpus_per_task=4, exclude="margpu009")


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
    #X["target"] = y
    # restrict X to 20K random samples
    if X.shape[0] > 20000:
        X = X.sample(20000, random_state=42)

    # restrict to 30 columns
    X = X.iloc[:, -30:]
    print(X.columns)
    normalization = hp_dic.pop("normalization", None)
    if normalization == "quantile":
        X = QuantileTransformer(n_quantiles=100, random_state=42).fit_transform(X)
    loader = GenericDataLoader(X)#, #target_column="target")

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

#run_model_on_dataset("forest_diffusion", tasks[1])


hps = {"normalization": "quantile"}

# with executor_gpu.batch():
#     for model in gpu_models:
#         for task_id in tasks:
#             #hp_dic = {"n_permutations": n_permutations, "n_ensembles": n_ensembles}
#             hp_dic = {}
#             hp_dic.update(hps)
#             executor_gpu.submit(run_model_on_dataset, model, task_id, hp_dic)

# with executor_cpu.batch():
#     #for model in ["gaussian_noise"]:
#         #for noise_std in [0.0000001]
#     for model in cpu_models:
#         for task_id in tasks:
#             #hp_dic = {"noise_std": noise_std}
#             hp_dic = {}
#             hp_dic.update(hps)
#             executor_cpu.submit(run_model_on_dataset, model, task_id, hp_dic)

if __name__ == "__main__":
    hp_dic = {}#{"n_batches": 10}
    score, score_train = run_model_on_dataset("tabpfn_points", tasks[0], hp_dic, save_results=False)
    print(score["tabpfn_points"])
    print(score["tabpfn_points"].columns)
    print(score["tabpfn_points"].index)
    print(score_train["tabpfn_points"])

