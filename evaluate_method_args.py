
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
import fire
import pandas as pd
import numpy as np
import json
import hashlib


def run_model_on_dataset(model_name, task_id, results_base_dir="results", normalization="quantile", save_results=True, **kwargs):
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
    # add target to X
    X["target"] = y
    # restrict X to 20K random samples
    if X.shape[0] > 20000:
        X = X.sample(20000, random_state=42)

    # restrict to 30 columns
    X = X.iloc[:, -30:]
    print(X.columns)
    if normalization == "quantile":
        X = QuantileTransformer(n_quantiles=100, random_state=42).set_output(transform="pandas").fit_transform(X)

    # take 1024 random rows from X
    indices = np.random.choice(X.index, 1024, replace=False)
    X_ref = X.loc[indices]
    X = X.drop(index=indices)

    loader = GenericDataLoader(X, target_column="target")
    loader_ref = GenericDataLoader(X_ref, target_column="target")

    print("Loaded")

    task_type = "regression"
    synthetic_size = 512


    score = Benchmarks.evaluate(
        [(model_name, model_name, hp_dic)],
        loader,
        X_ref=loader_ref,
        synthetic_size=synthetic_size,
        repeats=3,
        task_type=task_type,
        metrics = {
                    'sanity': ['data_mismatch', 'common_rows_proportion', 'close_values_probability', 'distant_values_probability',
                               'nearest_syn_neighbor_distance', "nearest_real_neighbor_distance",
                                "nearest_real_neighbor_distance_no_norm", "nearest_syn_neighbor_distance_no_norm",
                                "nearest_real_neighbor_distance_no_norm_on_train", "nearest_syn_neighbor_distance_no_norm_on_train",
                                "nearest_real_neighbor_distance_on_train", "nearest_syn_neighbor_distance_on_train"
                               ],
                    'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
                    'performance': ['mlp', 'xgb'],
                    'detection': ['detection_xgb', 'detection_mlp'],
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
    #score_train = pd.DataFrame(score_train)


    print("Benchmark done")
    if save_results:
        # save the results in results/dataset_id/model_name.pkl
        # create the folders if they don't exist
        os.makedirs(results_base_dir, exist_ok=True)
        os.makedirs(f"{results_base_dir}/{dataset_name}", exist_ok=True)
        os.makedirs(f"{results_base_dir}/{dataset_name}/{model_name}", exist_ok=True)
        config = {"model_name": model_name, "task_id": task_id, "normalization": normalization, "X_true_n_rows": X.shape[0], 
                  "X_true_n_cols": X.shape[1], "synthetic_size": synthetic_size, "task_type": task_type}
        config.update(hp_dic_original)
        # hash config
        config_hash = hashlib.sha256(pickle.dumps(config)).hexdigest()[:16]
        # create a new folder
        os.makedirs(f"{results_base_dir}/{dataset_name}/{model_name}/{config_hash}", exist_ok=True)
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
