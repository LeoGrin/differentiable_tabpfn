
import os
import pickle
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
from sklearn.preprocessing import LabelEncoder
import fire
import pandas as pd
import numpy as np
import json
import hashlib
from synthcity.plugins import Plugins



def run_model_on_dataset(model_name, task_id, n_synthetic_points=512, results_base_dir="results_classif", normalization="quantile", save_results=True, **kwargs):
    from synthcity_addons import generators
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    dataset_name = dataset.name
    print(f"Running {model_name} on {dataset_name}")
    # kwargs are the hyperparameters
    hp_dic = kwargs
    hp_dic.update({"strict": False})
    print("hp_dic", hp_dic)
    hp_dic_original = hp_dic.copy()
    hp_str = "_".join([f"{k}_{v}" for k, v in hp_dic.items()]) # before it's modified by synthcity
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    # restrict X to 20K random samples
    if X.shape[0] > 20000:
        X = X.sample(20000, random_state=42)

    print(X.columns)
    if normalization == "quantile":
        X = QuantileTransformer(n_quantiles=100, random_state=42).set_output(transform="pandas").fit_transform(X)

    # add target to X
    X["target"] = y
    # label encode target
    X["target"] = LabelEncoder().fit_transform(X["target"])

    # restrict to 30 columns
    X = X.iloc[:, -30:]
    print(X)
    # take 1024 random rows from X
    rng = np.random.RandomState(42)
    indices = rng.choice(X.index, 1024, replace=False)
    X_ref = X.loc[indices]
    X = X.drop(index=indices)

    loader = GenericDataLoader(X, target_column="target")
    loader_ref = GenericDataLoader(X_ref, target_column="target")

    print("Loaded")

    task_type = "classification"
    synthetic_size = n_synthetic_points

    config = {"model_name": model_name, "task_id": task_id, "normalization": normalization, "X_true_n_rows": X.shape[0], 
              "X_true_n_cols": X.shape[1], "synthetic_size": synthetic_size, "task_type": task_type}
    config.update(hp_dic_original)
    # hash config
    config_hash = hashlib.sha256(pickle.dumps(config)).hexdigest()[:16]
    if save_results or "tabpfn" in model_name:  
        os.makedirs(results_base_dir, exist_ok=True)
        os.makedirs(f"{results_base_dir}/{dataset_name}", exist_ok=True)
        os.makedirs(f"{results_base_dir}/{dataset_name}/{model_name}", exist_ok=True)
        os.makedirs(f"{results_base_dir}/{dataset_name}/{model_name}/{config_hash}", exist_ok=True)

    if "tabpfn" in model_name:
        hp_dic["store_animation_path"] = f"{results_base_dir}/{dataset_name}/{model_name}/{config_hash}/animation.mp4"
        hp_dic["store_intermediate_data"] = True


    #hp_dic["n_points_to_create"] = n_synthetic_points
    score = Benchmarks.evaluate(
        [(model_name, model_name, hp_dic)],
        loader,
        X_ref=loader_ref,
        synthetic_size=synthetic_size,
        repeats=6,
        task_type=task_type,
        verbose=100,
        metrics = {
                    'sanity': ['data_mismatch', 'common_rows_proportion', 'close_values_probability', 'distant_values_probability',
                               'nearest_syn_neighbor_distance', "nearest_real_neighbor_distance",
                                "nearest_real_neighbor_distance_no_norm", "nearest_syn_neighbor_distance_no_norm",
                                "nearest_real_neighbor_distance_no_norm_on_train", "nearest_syn_neighbor_distance_no_norm_on_train",
                                "nearest_real_neighbor_distance_on_train", "nearest_syn_neighbor_distance_on_train"
                               ],
                    'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
                    'performance': ['xgb', "mlp", "tabpfn", "mlp_bigger", "xgb_default"],
                    'detection': ['detection_xgb'],
                    'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score']
                }
    )[model_name]

    # score_train = Benchmarks.evaluate(
    #     [(model_name, model_name, hp_dic)],
    #     loader,
    #     loader,
    #     synthetic_size=synthetic_size,
    #     repeats=3,
    #     task_type=task_type
    # )[model_name]

    score = pd.DataFrame(score)

    if model_name in ["svm", "nu_svm"]:
        # do one pass just to get the number of points
        svm_plugin = Plugins().get(model_name, **hp_dic)#TODO all params
        svm_plugin.fit(loader)
        n_synthetic_points = svm_plugin.get_n_points_to_create()
        print(f"Number of points to create: {n_synthetic_points}")
        config["n_synthetic_points"] = n_synthetic_points
    #score_train = pd.DataFrame(score_train)


    print("Benchmark done")
    if save_results:
        # save the results in results/dataset_id/model_name.pkl
        # create the folders if they don't exist
        # save the config
        # with open(f"{results_base_dir}/{dataset_name}/{config_hash}/config.pkl", "wb") as f:
        #     pickle.dump(config, f)
        # save config as json
        with open(f"{results_base_dir}/{dataset_name}/{model_name}/{config_hash}/config.json", "w") as f:
            json.dump(config, f)
        # save the scores
        score.to_csv(f"{results_base_dir}/{dataset_name}/{model_name}/{config_hash}/scores.csv")
        #score_train.to_csv(f"{results_base_dir}/{dataset_name}/{model_name}/{config_hash}/scores_train.csv")
    else:
        return score#, score_train


if __name__ == "__main__":
    print(fire.Fire(run_model_on_dataset))
