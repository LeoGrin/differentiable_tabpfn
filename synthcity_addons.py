# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.distribution import (
    Distribution,
    IntegerDistribution,
)
from synthcity.plugins.generic.plugin_dummy_sampler import DummySamplerPlugin
from tabpfn import TabPFNClassifier
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from ForestDiffusion import ForestDiffusionModel
from synthcity.plugins import Plugins
from smote import MySMOTE
from imblearn.over_sampling import SMOTE
from utils import create_animation
from modules import ResNet, MLP
import ot
from kmean_tabpfn import KmeanTabPFNClassifier
from sklearn.cluster import KMeans, HDBSCAN

class gaussian_noise_plugin(Plugin):
    """Gaussian Noise integration in synthcity."""

    def __init__(
        self,
        noise_std: float = 0.1,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.noise_std = noise_std

    @staticmethod
    def name() -> str:
        return "gaussian_noise"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        We can customize the hyperparameter space, and use it in AutoML benchmarks.
        """
        #TODO
        return [
            IntegerDistribution(name="embedding_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="batch_size", low=100, high=300, step=50),
            IntegerDistribution(name="n_iter", low=100, high=500, step=50),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "gaussian_noise_plugin":
        self.X = X.numpy()
        print("fit")
        return self

    def sample(self, count: int, **kwargs: Any) -> pd.DataFrame:
        if count > len(self.X):
            raise ValueError("Requested count exceeds the available data.")
        indices = np.random.choice(len(self.X), count, replace=False)
        selected_points = self.X[indices]
        noise = np.random.normal(0., self.noise_std, selected_points.shape)
        noisy_points = selected_points + noise
        print("sample", count, len(noisy_points))
        
        return noisy_points
    
    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.sample, count, syn_schema)


class oracle_plugin(Plugin):
    """Subsample a reference sample from the real data. The reference dataset should be different from the train and test
    datasets."""

    def __init__(
        self,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        

    @staticmethod
    def name() -> str:
        return "oracle"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        We can customize the hyperparameter space, and use it in AutoML benchmarks.
        """
        #TODO
        return [
            IntegerDistribution(name="embedding_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="batch_size", low=100, high=300, step=50),
            IntegerDistribution(name="n_iter", low=100, high=500, step=50),
        ]

    def _fit(self, X: DataLoader, X_ref: DataLoader, *args: Any, **kwargs: Any) -> "oracle_plugin":
        self.X = X_ref.dataframe()
        return self

    
    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        assert self.strict == False, "Oracle plugin does not support strict mode"
        def _sample(count: int) -> pd.DataFrame:
            baseline = self.X
            #constraints = syn_schema.as_constraints()

            #baseline = constraints.match(baseline)
            if len(baseline) == 0:
                raise ValueError("Cannot generate data")

            if len(baseline) <= count:
                return baseline.sample(frac=1)

            return baseline.sample(count, replace=False).reset_index(drop=True)

        return self._safe_generate(_sample, count, syn_schema)



class tabpfn_points_plugin(Plugin):
    """TabPFN integration in synthcity."""

    def __init__(
        self,
        n_random_test_samples: int = 3_000,
        device: str = "cuda:0",
        n_batches: int = 200,
        lr: float = 0.1,
        n_permutations: int = 3,
        n_ensembles: int = 3,
        initialization_strategy: str = "uniform",
        store_intermediate_data: bool = False,
        store_animation_path: bool = None,
        store_false_data_path: bool = None,
        n_test_from_false_train: int = 0,
        n_random_features_to_add: int = 1,
        random_test_points_scale: float = 2,
        init_scale_factor: float = 5,
        noise_std: float = 0.1,
        preprocessing: str = "standard",
        loss: str = "individual",
        use_kmeans_tabpfn: bool = False,
        n_clusters_kmeans_tabpfn: int = 3,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.n_random_test_samples = n_random_test_samples
        self.device = device
        self.n_batches = n_batches
        self.lr = lr
        self.n_permutations = n_permutations
        self.n_ensembles = n_ensembles
        if preprocessing == "standard":
            self.preprocessor = StandardScaler()
        elif preprocessing == "none":
            self.preprocessor = None
        else:
            raise ValueError(f"Preprocessing {preprocessing} not supported")
        self.store_intermediate_data = store_intermediate_data
        self.store_animation_path = store_animation_path
        self.store_false_data_path = store_false_data_path
        self.n_test_from_false_train = n_test_from_false_train
        self.n_random_features_to_add = n_random_features_to_add
        self.random_test_points_scale = random_test_points_scale
        if initialization_strategy == "gaussian_noise":
            self.initialization_strategy =  Plugins().get("gaussian_noise", strict=False, noise_std=noise_std)
        elif initialization_strategy == "smote":
            self.initialization_strategy =  Plugins().get("smote", strict=False)
        else:   
            self.initialization_strategy = initialization_strategy
        self.init_scale_factor = init_scale_factor
        self.noise_std = noise_std
        self.loss = loss
        self.use_kmeans_tabpfn = use_kmeans_tabpfn
        self.n_clusters_kmeans_tabpfn = n_clusters_kmeans_tabpfn
        if store_animation_path is not None:
            assert store_intermediate_data, "store_intermediate_data must be True to store animation data"
        if store_intermediate_data:
            self.loss_list = []
            self.all_X_false_train = []
            self.accuracy_list = []


    @staticmethod
    def name() -> str:
        return "tabpfn_points"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        We can customize the hyperparameter space, and use it in AutoML benchmarks.
        """
        #TODO
        return [
            IntegerDistribution(name="embedding_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="batch_size", low=100, high=300, step=50),
            IntegerDistribution(name="n_iter", low=100, high=500, step=50),
        ]

    def _fit(self, X: DataLoader, X_false_train_init=None, *args: Any, **kwargs: Any) -> "tabpfn_points_plugin":
        if self.preprocessor is not None:
            self.X_true = self.preprocessor.fit_transform(X.numpy()) # all numerical features for now
        else:
            self.X_true = X.numpy()
        if X_false_train_init is None:
            if self.initialization_strategy == "uniform":
                X_false_train = (np.random.rand(512, X.shape[1]) * 2 * self.random_test_points_scale - self.random_test_points_scale) / self.init_scale_factor
            elif self.initialization_strategy == "gaussian_noise":
                #TODO use plugin to generate noise
                indices = np.random.choice(X.shape[0], 512, replace=False)
                X_false_train = self.X_true[indices]
                noise = np.random.normal(0, self.noise_std, X_false_train.shape)
                X_false_train += noise
            elif type(self.initialization_strategy) != str:
                # in this case it should be a plugin
                self.initialization_strategy.fit(pd.DataFrame(self.X_true))
                X_false_train = self.initialization_strategy.generate(512).numpy()
            else:
                raise ValueError(f"Initialization strategy {self.initialization_strategy} not supported")
        else:
            X_false_train = X_false_train_init
        self.X_false_train = torch.tensor(X_false_train).float().to(self.device)
        self.X_false_train.requires_grad = True
        self.X_true = torch.tensor(self.X_true, dtype=torch.float32).to(self.device)
        X_random_test = np.random.rand(self.n_random_test_samples, self.X_true.shape[1]) * 2 * self.random_test_points_scale - self.random_test_points_scale
        X_random_test = torch.tensor(X_random_test).float().to(self.device)

        
        optimizer = torch.optim.Adam([self.X_false_train], lr=self.lr)

        tabpfn_classifier = TabPFNClassifier(device=self.device, N_ensemble_configurations=self.n_permutations,
                                              no_preprocess_mode=True, no_grad=False, normalize=False) #TODO
        # if self.use_kmeans_tabpfn:
        #     tabpfn_classifier = KmeanTabPFNClassifier(n_clusters=self.n_clusters_kmeans_tabpfn, base_classifier=tabpfn_classifier)
        
        if self.store_intermediate_data:
            self.all_X_false_train.append(self.X_false_train.detach().cpu().numpy().copy())

        for batch in tqdm(range(self.n_batches)):
            n_train = 512 #TODO
            n_test = min(2048, self.X_true.shape[0])
            #tabpfn_output_proba_list = []
            loss = torch.tensor(0.0, device=self.device)
            for _ in range(self.n_ensembles):
                indices_train = np.random.choice(self.X_true.shape[0], n_train, replace=False)
                X_batch_train = self.X_true[indices_train]
                indices_test = np.random.choice(self.X_true.shape[0], n_test, replace=False)
                X_batch_test = self.X_true[indices_test]
                #indices_false_test = np.random.choice(X_random_test.shape[0], len(X_batch_test), replace=False)
                #X_false_test = X_random_test[indices_false_test]
                indices_false_test_from_random = np.random.choice(X_random_test.shape[0], len(X_batch_test) - self.n_test_from_false_train, replace=False)
                X_false_test_from_random = X_random_test[indices_false_test_from_random]
                indices_false_test_from_false_train = np.random.choice(self.X_false_train.shape[0], self.n_test_from_false_train, replace=False)
                X_false_test_from_false_train = self.X_false_train.detach()[indices_false_test_from_false_train]
                X_false_test = torch.cat((X_false_test_from_random, X_false_test_from_false_train), dim=0)

                indices_false_train = np.random.choice(self.X_false_train.shape[0], min(len(X_batch_train), self.X_false_train.shape[0]), replace=False) #TODO
                X_false_batch_train = self.X_false_train[indices_false_train]

                X_train = torch.cat((X_batch_train, X_false_batch_train), dim=0)
                X_test = torch.cat((X_batch_test, X_false_test), dim=0)
                y_train = torch.cat((torch.ones(X_batch_train.shape[0]), torch.zeros(X_false_batch_train.shape[0])), dim=0).long()
                y_test = torch.cat((torch.ones(X_batch_test.shape[0]), torch.zeros(X_false_test.shape[0])), dim=0).long()
                #y_test = torch.ones(X_batch_test.shape[0]).long()
                #y_test = torch.zeros(X_false_test.shape[0]).long()

                perm_train = torch.randperm(X_train.shape[0])
                X_train = X_train[perm_train]
                y_train = y_train[perm_train]
                perm_test = torch.randperm(X_test.shape[0])
                X_test = X_test[perm_test]
                y_test = y_test[perm_test]


                # Use kmeans to cluster X_train
                # kmeans = KMeans(n_clusters=self.n_clusters_kmeans_tabpfn, random_state=42)
                # kmeans.fit(X_train.detach().cpu().numpy())
                # cluster_labels = kmeans.labels_
                # test_cluster_labels = kmeans.predict(X_test.cpu().numpy())
                hdbscan = HDBSCAN(min_cluster_size=10)
                X_combined = torch.cat((X_train, X_test), dim=0).detach().cpu().numpy()
                hdbscan.fit(X_combined)
                cluster_labels = hdbscan.labels_[:X_train.shape[0]]
                test_cluster_labels = hdbscan.labels_[X_train.shape[0]:]
                for i in range(self.n_clusters_kmeans_tabpfn):
                    print(f"Cluster {i}: {len(cluster_labels[cluster_labels == i])}")
                    X_train_cluster_i = X_train[cluster_labels == i]
                    y_train_cluster_i = y_train[cluster_labels == i]
                    print(f"Cluster {i}: {np.unique(y_train_cluster_i.cpu().numpy(), return_counts=True)}")
                    class_counts = np.bincount(y_train_cluster_i.cpu().numpy())
                    if len(class_counts) < 2 or min(class_counts) < 5:
                        print(f"Cluster {i} has less than 5 samples")
                        continue
                    X_test_cluster_i = X_test[test_cluster_labels == i]
                    y_test_cluster_i = y_test[test_cluster_labels == i]
                    # add a third feature to X_train and X_test with random values
                    if self.n_random_features_to_add > 0:
                        X_train_cluster_i = torch.cat((X_train_cluster_i, torch.randn(X_train_cluster_i.shape[0], self.n_random_features_to_add).to(self.device)), dim=1)
                        X_test_cluster_i = torch.cat((X_test_cluster_i, torch.randn(X_test_cluster_i.shape[0], self.n_random_features_to_add).to(self.device)), dim=1)
                    tabpfn_classifier.fit(X_train_cluster_i, y_train_cluster_i, overwrite_warning=True)
                    tabpfn_output_proba_cluster_i = tabpfn_classifier.predict_proba(X_test_cluster_i)
                    chance_proba = torch.mean(y_train_cluster_i.float())
                    print(f"Chance proba: {chance_proba}")
                    print(tabpfn_output_proba_cluster_i[:10, 1])
                    loss +=  torch.mean((tabpfn_output_proba_cluster_i[:, 1] - chance_proba)**2)
                #tabpfn_output_proba_list.append(tabpfn_output_proba)
            #tabpfn_output_proba = torch.stack(tabpfn_output_proba_list).mean(dim=0)


            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                print(f"Batch {batch} loss: {loss.item()}")

            if self.store_intermediate_data:
                self.loss_list.append(loss.item())
                self.all_X_false_train.append(self.X_false_train.detach().cpu().numpy().copy())
                #y_pred = tabpfn_output_proba.argmax(dim=1)
                #accuracy = torch.mean((y_pred.detach().cpu() == y_test).float()).item()
               # self.accuracy_list.append(accuracy)

        if self.store_animation_path is not None:
            # create an animation of the false train points
            create_animation(self.all_X_false_train, X_batch_train.detach().cpu().numpy(), self.store_animation_path,
                                step=max(self.n_batches // 20, 5))
            
        if self.store_false_data_path is not None:
            np.save(self.store_false_data_path, self.X_false_train.detach().cpu().numpy())
    
        return self.X_false_train.detach().cpu(), y_test.detach().cpu()

    def sample(self, count: int, **kwargs: Any) -> pd.DataFrame:
        if count > len(self.X_false_train):
            raise ValueError("Requested count exceeds the available data.")
        indices = np.random.choice(len(self.X_false_train), count, replace=False)
        false_points = self.X_false_train[indices].detach().cpu().numpy()
        if self.preprocessor is not None:
            return self.preprocessor.inverse_transform(false_points)
        else:
            return false_points
    
    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.sample, count, syn_schema)
    


class tabpfn_generator_plugin(Plugin):
    """TabPFN integration in synthcity."""

    def __init__(
        self,
        n_random_test_samples: int = 3_000,
        device: str = "cuda:0",
        n_batches: int = 200,
        lr: float = 0.1,
        n_permutations: int = 3,
        n_ensembles: int = 3,
        initialization_strategy: str = "uniform",
        store_intermediate_data: bool = False,
        store_animation_path: bool = None,
        store_false_data_path: bool = None,
        n_test_from_false_train: int = 0,
        n_random_features_to_add: int = 1,
        random_test_points_scale: float = 2,
        init_scale_factor: float = 5,
        noise_std: float = 0.1,
        preprocessing: str = "standard",
        use_wasserstein: bool = False,
        model: str = "mlp",
        n_layers: int = 3,
        d_hidden: int = 256,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.n_random_test_samples = n_random_test_samples
        self.device = device
        self.n_batches = n_batches
        self.lr = lr
        self.n_permutations = n_permutations
        self.n_ensembles = n_ensembles
        if preprocessing == "standard":
            self.preprocessor = StandardScaler()
        elif preprocessing == "none":
            self.preprocessor = None
        else:
            raise ValueError(f"Preprocessing {preprocessing} not supported")
        self.store_intermediate_data = store_intermediate_data
        self.store_animation_path = store_animation_path
        self.store_false_data_path = store_false_data_path
        self.n_test_from_false_train = n_test_from_false_train
        self.n_random_features_to_add = n_random_features_to_add
        self.random_test_points_scale = random_test_points_scale
        self.initialization_strategy = initialization_strategy
        self.init_scale_factor = init_scale_factor
        self.noise_std = noise_std
        self.use_wasserstein = use_wasserstein
        self.model = model
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        if store_animation_path is not None:
            assert store_intermediate_data, "store_intermediate_data must be True to store animation data"
        if store_intermediate_data:
            self.loss_list = []
            self.all_X_false_train = []
            self.accuracy_list = []


    @staticmethod
    def name() -> str:
        return "tabpfn_generator"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        We can customize the hyperparameter space, and use it in AutoML benchmarks.
        """
        #TODO
        return [
            IntegerDistribution(name="embedding_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="batch_size", low=100, high=300, step=50),
            IntegerDistribution(name="n_iter", low=100, high=500, step=50),
        ]

    def _fit(self, X: DataLoader, X_false_train_init=None, *args: Any, **kwargs: Any) -> "tabpfn_points_plugin":
        if self.preprocessor is not None:
            X_true = self.preprocessor.fit_transform(X.numpy()) # all numerical features for now
        else:
            X_true = X.numpy()
        X_true = torch.tensor(X_true, dtype=torch.float32).to(self.device)
        X_random_test = np.random.rand(self.n_random_test_samples, X.shape[1]) * 2 * self.random_test_points_scale - self.random_test_points_scale
        X_random_test = torch.tensor(X_random_test).float().to(self.device)

        if self.model == "mlp": 
            self.mlp_config = {"d_in": X_true.shape[1], "d_out": X_true.shape[1], "d_layers": [self.d_hidden]*self.n_layers, "dropout": 0.1}
            self.generator_model = MLP.make_baseline(**self.mlp_config).to(self.device)
        elif self.model == "resnet":
            self.resnet_config = {"d_in": X_true.shape[1],
                                   "d_out": X_true.shape[1], 
                                   "n_blocks": self.n_layers, "d_main": self.d_hidden, "d_hidden": 2 * self.d_hidden, "dropout_first": 0.1, "dropout_second": 0.1}
            self.generator_model = ResNet.make_baseline(**self.resnet_config).to(self.device)

        
        optimizer = torch.optim.Adam(self.generator_model.parameters(), lr=self.lr)

        tabpfn_classifier = TabPFNClassifier(device=self.device, N_ensemble_configurations=self.n_permutations,
                                              no_preprocess_mode=True, no_grad=False, normalize=False)
        
        #if self.store_intermediate_data:
        #self.all_X_false_train.append(self.X_false_train.detach().cpu().numpy())
        n_train = 512
        n_test = min(2048, X_true.shape[0])

        for batch in tqdm(range(self.n_batches)):
            if not self.use_wasserstein:
                tabpfn_output_proba_list = []
                for _ in range(self.n_ensembles):
                    indices_train = np.random.choice(X_true.shape[0], n_train, replace=False)
                    X_batch_train = X_true[indices_train]
                    indices_test = np.random.choice(X_true.shape[0], n_test, replace=False)
                    X_batch_test = X_true[indices_test]
                    #indices_false_test = np.random.choice(X_random_test.shape[0], len(X_batch_test), replace=False)
                    #X_false_test = X_random_test[indices_false_test]
                    indices_false_test_from_random = np.random.choice(X_random_test.shape[0], len(X_batch_test), replace=False)
                    X_false_test_from_random = X_random_test[indices_false_test_from_random]
                    #indices_false_test_from_false_train = np.random.choice(512, self.n_test_from_false_train, replace=False)
                    #X_false_test_from_false_train = self.X_false_train.detach()[indices_false_test_from_false_train]
                    X_false_test = X_false_test_from_random#torch.cat((X_false_test_from_random, X_false_test_from_false_train), dim=0) #TODO

                    #indices_false_train = np.random.choice(self.X_false_train.shape[0], len(X_batch_train), replace=False)
                    #X_false_batch_train = self.X_false_train[indices_false_train]
                    # sample noise
                    noise = torch.randn(X_batch_train.shape[0], X_batch_train.shape[1]).to(self.device)
                    noise.requires_grad = True
                    X_false_batch_train = self.generator_model(noise)

                    X_train = torch.cat((X_batch_train, X_false_batch_train), dim=0)
                    X_test = torch.cat((X_batch_test, X_false_test), dim=0)
                    y_train = torch.cat((torch.ones(X_batch_train.shape[0]), torch.zeros(X_false_batch_train.shape[0])), dim=0).long()
                    y_test = torch.cat((torch.ones(X_batch_test.shape[0]), torch.zeros(X_false_test.shape[0])), dim=0).long()
                    #y_test = torch.ones(X_batch_test.shape[0]).long()
                    #y_test = torch.zeros(X_false_test.shape[0]).long()

                    perm_train = torch.randperm(X_train.shape[0])
                    X_train = X_train[perm_train]
                    y_train = y_train[perm_train]
                    perm_test = torch.randperm(X_test.shape[0])
                    X_test = X_test[perm_test]
                    y_test = y_test[perm_test]

                    # add a third feature to X_train and X_test with random values
                    if self.n_random_features_to_add > 0:
                        X_train = torch.cat((X_train, torch.randn(X_train.shape[0], self.n_random_features_to_add).to(self.device)), dim=1)
                        X_test = torch.cat((X_test, torch.randn(X_test.shape[0], self.n_random_features_to_add).to(self.device)), dim=1)
                    tabpfn_classifier.fit(X_train, y_train, overwrite_warning=True)
                    tabpfn_output_proba = tabpfn_classifier.predict_proba(X_test)
                    tabpfn_output_proba_list.append(tabpfn_output_proba)
                tabpfn_output_proba = torch.stack(tabpfn_output_proba_list).mean(dim=0)

                loss = torch.mean(torch.abs((tabpfn_output_proba[:, 0] - tabpfn_output_proba[:, 1]))**2)
                #loss = (torch.mean(tabpfn_output_proba[:, 0]) - torch.mean(tabpfn_output_proba[:, 1]))**2 #TODO
                #loss = (torch.mean(tabpfn_output_proba[y_test == 0, 0]) - torch.mean(tabpfn_output_proba[y_test == 1, 0]))**2
            elif self.use_wasserstein:
                noise = torch.randn(n_train, X_true.shape[1]).to(self.device)
                noise.requires_grad = True
                X_false_batch_train = self.generator_model(noise)
                # Compute the Wasserstein distance using the Sinkhorn algorithm
                M = ot.dist(X_true, X_false_batch_train, metric='euclidean')
                a = torch.ones((X_true.shape[0],), device=self.device) / X_true.shape[0]
                b = torch.ones((X_false_batch_train.shape[0],), device=self.device) / X_false_batch_train.shape[0]
                reg = 1e-3  # Regularization parameter for Sinkhorn algorithm
                loss = ot.sinkhorn2(a, b, M, reg)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                print(f"Batch {batch} loss: {loss.item()}")

            if self.store_intermediate_data:
                self.loss_list.append(loss.item())
                #self.all_X_false_train.append(self.X_false_train.detach().cpu().numpy())
                self.all_X_false_train.append(X_false_batch_train.detach().cpu().numpy())
                if not self.use_wasserstein:
                    y_pred = tabpfn_output_proba.argmax(dim=1)
                    accuracy = torch.mean((y_pred.detach().cpu() == y_test).float()).item()
                    self.accuracy_list.append(accuracy)

        if self.store_animation_path is not None:
            # create an animation of the false train points
            if not self.use_wasserstein:
                create_animation(self.all_X_false_train, X_batch_train.detach().cpu().numpy(), self.store_animation_path,
                                    step=max(self.n_batches // 20, 5))
            else:
                create_animation(self.all_X_false_train, X_true.detach().cpu().numpy(), self.store_animation_path,
                                    step=max(self.n_batches // 20, 5))

        if self.store_false_data_path is not None:
            np.save(self.store_false_data_path, X_false_batch_train.detach().cpu().numpy())
        if not self.use_wasserstein:
            return tabpfn_output_proba.detach().cpu(), X_false_batch_train.detach().cpu(), y_test.detach().cpu()
        else:
            return None

    def sample(self, count: int, **kwargs: Any) -> pd.DataFrame:
        noise = torch.randn(count, self.mlp_config["d_in"]).to(self.device)
        #noise.requires_grad = True
        self.generator_model.eval()
        X_false_batch_train = self.generator_model(noise)
        self.generator_model.train()
        return X_false_batch_train.detach().cpu().numpy()
    
    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.sample, count, syn_schema)
    

class forest_diffusion_plugin(Plugin):
    """ForestDiffusion integration in synthcity."""

    def __init__(
        self,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "forest_diffusion"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        We can customize the hyperparameter space, and use it in AutoML benchmarks.
        """
        #TODO
        return [
            IntegerDistribution(name="embedding_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="batch_size", low=100, high=300, step=50),
            IntegerDistribution(name="n_iter", low=100, high=500, step=50),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "forest_diffusion_plugin":
        self.model = ForestDiffusionModel(X.numpy(), label_y=None, n_batch=1, n_t=15, duplicate_K=100, bin_indexes=[], cat_indexes=[], int_indexes=[], diffusion_type='flow', n_jobs=-1)
        return self

    def sample(self, count: int, **kwargs: Any) -> pd.DataFrame:
        return self.model.generate(batch_size=count)
    
    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.sample, count, syn_schema)
    


import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


class smote_plugin(Plugin):
    """SMOTE integration in synthcity.
    THIS IS VERY SIMPLIFIED VERSION OF SMOTE, JUST FOR DEMONSTRATION PURPOSES.
    I should use the imbalanced-learn library for a more complete implementation."""

    def __init__(
        self,
        k_neighbors: int = 5,
        lam_min: float = 0.0,
        lam_max: float = 1.0,
        add_noise_std: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.k_neighbors = k_neighbors
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.add_noise_std = add_noise_std

    @staticmethod
    def name() -> str:
        return "smote"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        We can customize the hyperparameter space, and use it in AutoML benchmarks.
        """
        #TODO
        return [
            IntegerDistribution(name="embedding_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="batch_size", low=100, high=300, step=50),
            IntegerDistribution(name="n_iter", low=100, high=500, step=50),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "smote_plugin":
        # Fit the NearestNeighbors model
        self.X = X.numpy()
        self.nn = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn.fit(self.X)

    def sample(self, count: int, **kwargs: Any) -> pd.DataFrame:
        synthetic_samples = []
        for _ in range(count):
            # Randomly choose a sample from X
            idx = np.random.randint(len(self.X))
            sample = self.X[idx]
            
            # Find the k nearest neighbors
            neighbors = self.nn.kneighbors([sample], return_distance=False)
            
            # Randomly choose one of the neighbors
            neighbor_idx = np.random.choice(neighbors[0])
            neighbor = self.X[neighbor_idx]
            
            # Generate a synthetic sample by interpolation
            lam = np.random.uniform(self.lam_min, self.lam_max)
            synthetic_sample = sample + lam * (neighbor - sample)
            synthetic_sample += np.random.randn(synthetic_sample.shape[0]) * self.add_noise_std
            synthetic_samples.append(synthetic_sample)
    
        return np.array(synthetic_samples)
    
    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.sample, count, syn_schema)



class smote_imblearn_plugin(Plugin):
    """SMOTE integration using imbalanced-learn in synthcity."""
    #TODO more parameters

    def __init__(
        self,
        k_neighbors: int = 5,
        add_noise_std: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.k_neighbors = k_neighbors
        self.add_noise_std = add_noise_std

    @staticmethod
    def name() -> str:
        return "smote_imblearn"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        """
        We can customize the hyperparameter space, and use it in AutoML benchmarks.
        """
        #TODO
        return [
            IntegerDistribution(name="embedding_n_units", low=100, high=500, step=50),
            IntegerDistribution(name="batch_size", low=100, high=300, step=50),
            IntegerDistribution(name="n_iter", low=100, high=500, step=50),
        ]

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> "smote_imblearn_plugin":
        self.X = X.numpy()
        self.smote = SMOTE(k_neighbors=self.k_neighbors)
        return self

    def sample(self, count: int, **kwargs: Any) -> pd.DataFrame:
        # Create a fake y to use SMOTE to generate synthetic samples
        X_all = np.vstack([self.X, np.random.rand(count, self.X.shape[1])])
        y = np.zeros(len(X_all))
        y[-len(self.X) + count:] = 1
        
        X_res, y_res = self.smote.fit_resample(X_all, y)
        X_false = X_res[-count:]
        X_false += np.random.randn(X_false.shape[0], X_false.shape[1]) * self.add_noise_std
        return X_false
    
    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.sample, count, syn_schema)

generators = Plugins()

generators.add("tabpfn_points", tabpfn_points_plugin)
generators.add("tabpfn_generator", tabpfn_generator_plugin)
generators.add("forest_diffusion", forest_diffusion_plugin)
generators.add("smote", smote_plugin)
generators.add("smote_imblearn", smote_imblearn_plugin)
generators.add("gaussian_noise", gaussian_noise_plugin)
generators.add("oracle", oracle_plugin)

