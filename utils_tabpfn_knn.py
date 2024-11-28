import numpy as np
import os
import faiss
# from faiss import StandardGpuResources
import random
import torch
import torch.nn as nn
import shutil
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def fix_missing(X_nni, y_nni, sizes_per_class, indices_X_nni, y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    missing = False
    for c, count in zip(classes, counts):
        if count < sizes_per_class[c]:
            missing = True
            break
    if not missing:
        return X_nni, y_nni

    new_X_nni, new_y_nni = [], []
    offset = 0
    for c, size in enumerate(sizes_per_class):
        missing_indices = np.where(indices_X_nni[c][0] == -1)[0]
        if len(missing_indices) > 0:
            miss_idx = missing_indices[0]
        else:
            miss_idx = size
        new_X_nni.append(X_nni[offset : offset + miss_idx])
        new_y_nni.append(y_nni[offset : offset + miss_idx])
        offset += size
    X_nni = np.concatenate(new_X_nni, axis=0)
    y_nni = np.concatenate(new_y_nni)

    return X_nni, y_nni


def create_dataloaders(args, data):
    for split in ["train", "valid", "test"]:
        X, y = data["X_" + split], data["y_" + split]
        X_one_hot = data["X_" + split + "_one_hot"]
        if (args.method == "ft" and args.type == "ftknn") or (args.method == "knn"):
            data[split + "_loader"] = DataLoader(
                TensorDataset(
                    torch.Tensor(X), torch.Tensor(y), torch.Tensor(X_one_hot)
                ),
                batch_size=args.batch_size if split == "train" else args.batch_size_inf,
                shuffle=True if split == "train" else False,
                drop_last=False,
                num_workers=0,
            )
        else:
            if split == "train":
                data[split + "_loader"] = DataLoader(
                    TensorDataset(torch.Tensor(X), torch.Tensor(y)),
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0,
                )
            else:
                data[split + "_loader"] = DataLoader(
                    TensorDataset(torch.Tensor(X), torch.Tensor(y)),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                )


def clone_linear_layer(layer, device="cpu"):
    if not isinstance(layer, nn.Linear):
        raise ValueError("Provided layer must be an instance of torch.nn.Linear")

    cloned_layer = nn.Linear(
        layer.in_features, layer.out_features, bias=(layer.bias is not None)
    )
    cloned_layer.weight.data = layer.weight.data.to(device).clone()
    if layer.bias is not None:
        cloned_layer.bias.data = layer.bias.data.to(device).clone()

    return cloned_layer


def save_numpy(file_path, x):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, x)


def pad_x(X, num_features=100):
    seq_len, batch_size, n_features = X.shape
    zero_feature_padding = torch.zeros(
        (seq_len, batch_size, num_features - n_features), device=X.device
    )
    return torch.cat([X, zero_feature_padding], -1)


def compute_metrics(y_target, y_pred):
    acc = accuracy_score(y_target, y_pred.argmax(1))
    f1 = f1_score(y_target, y_pred.argmax(1), average="weighted")

    if len(np.unique(y_target)) == 2:
        auc = roc_auc_score(y_target, y_pred[:, 1])
    else:
        auc = roc_auc_score(y_target, y_pred, multi_class="ovo")

    return acc, f1, auc


def get_sizes_per_class(args, y_train, num_classes, context_length=None):
    assert num_classes <= 10, "can only handle up to 10 classes"

    if not context_length:
        context_length = args.context_length

    if args.class_choice == "equal":
        min_num_items = 2
        sizes_per_class = [
            max(int(context_length * sum(y_train == c) / len(y_train)), min_num_items)
            for c in range(num_classes)
        ]

        # add the remaining in case of leftovers
        residual = context_length - sum(sizes_per_class)
        if residual <= 0:
            max_idx = np.argmax(np.array(sizes_per_class))
            sizes_per_class[max_idx] += residual
        else:
            min_idx = np.argmax(np.array(sizes_per_class))
            sizes_per_class[min_idx] += residual
    elif args.class_choice == "balance":
        sizes_per_class = [context_length // num_classes] * num_classes
        # add the remaining in case of leftovers
        sizes_per_class[0] += context_length - sum(sizes_per_class)
    else:
        raise NotImplementedError

    return sizes_per_class


def setup_experiment(args):
    # Create directories and files
    experiment_path = f"results/{args.datasets}/{args.exp_name}"

    if os.path.exists(experiment_path):
        # Remove the folder and then recreate it
        print("Folder exists, removing its content and recreating it.")
        shutil.rmtree(experiment_path)
        os.makedirs(experiment_path)
        os.makedirs(experiment_path + "/data")
    else:
        print("Folder does not exist, creating it.")
        os.makedirs(experiment_path)
        os.makedirs(experiment_path + "/data")

    # Initialize writers and files
    writer = SummaryWriter(experiment_path)
    results_file_path = os.path.join(experiment_path, "results.csv")
    results_file = open(results_file_path, "w")

    # Seed for reproducibility
    seed_everything(args.seed)

    # Write setups to the results file
    for arg, value in vars(args).items():
        results_file.write(f"{arg},{value}\n")
    results_file.flush()

    return writer, results_file, experiment_path


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class MulticlassFaiss:
    def __init__(self, embX, X_orig, y, metric="L2", gpu_id=0, sizes_per_class=None):
        assert isinstance(embX, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        if embX.dtype != np.float32:
            embX = embX.astype(np.float32)

        self.X = X_orig
        self.y = y

        self.sizes_per_class = sizes_per_class

        self.n_classes = len(np.unique(y))
        # self.gpu_resources = StandardGpuResources()
        if metric == "L2":
            self.indexes = [
                faiss.IndexFlatL2(embX.shape[1]) for _ in range(self.n_classes)
            ]
        elif metric == "IP":
            self.indexes = [
                faiss.IndexFlatIP(embX.shape[1]) for _ in range(self.n_classes)
            ]
        else:
            raise NotImplementedError

        # self.indexes = [faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, index) for index in self.indexes]
        for index, Xs in zip(
            self.indexes, [embX[y == i] for i in range(self.n_classes)]
        ):
            index.add(Xs)

    def get_knn_indices(self, queries, ks):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()

        if isinstance(ks, int):
            ks = [ks] * self.n_classes
        assert (
            len(ks) == self.n_classes
        ), "ks must have the same length as the number of classes"

        knns = [index.search(queries, k) for index, k in zip(self.indexes, ks)]
        indices_Xs = [x[1] for x in knns]
        ys = np.concatenate([np.ones(k) * i for i, k in enumerate(ks)])
        return indices_Xs, ys

    def get_knn(self, queries, ks=None):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = queries.astype(np.float32)

        if ks is None:
            assert self.sizes_per_class is not None
            ks = self.sizes_per_class

        if isinstance(ks, int):
            ks = [ks] * self.n_classes
        assert (
            len(ks) == self.n_classes
        ), "ks must have the same length as the number of classes"

        knns = [index.search(queries, k) for index, k in zip(self.indexes, ks)]
        indices_Xs = [x[1] for x in knns]
        Xs = [self.X[self.y == i][indices] for i, indices in enumerate(indices_Xs)]
        Xs = np.concatenate(Xs, axis=1)
        ys = [self.y[self.y == i][indices] for i, indices in enumerate(indices_Xs)]
        ys = np.concatenate(ys, axis=1)
        distances = [x[0] for x in knns]
        distances = np.concatenate(distances, axis=1)

        # because TabPFN is seq len first, batch size second...
        swap01 = lambda x: np.swapaxes(x, 0, 1)
        return swap01(Xs), swap01(ys), swap01(distances), indices_Xs


class SingleclassFaiss:
    def __init__(self, X, y, metric="L2", gpu_id=0):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        self.y = y

        # self.gpu_resources = StandardGpuResources()
        if metric == "L2":
            self.index = faiss.IndexFlatL2(X.shape[1])
        elif metric == "IP":
            self.index = faiss.IndexFlatIP(X.shape[1])
        else:
            raise NotImplementedError

        # self.index = faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, self.index)
        self.index.add(X)

    def get_knn_indices(self, queries, k):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        assert isinstance(k, int)

        knns = self.index.search(queries, k)
        indices_Xs = knns[1]
        ys = self.y[indices_Xs]
        return indices_Xs, ys



@torch.no_grad()
def compute_embedding(
    model,
    data,
    X,
    y=None,
    is_context=False,
    embedding_type="raw",
    device="cuda:0",
    embedding_layer=None,
):
    if embedding_type == "raw":
        return X
    elif embedding_type == "logits":
        num_features = X.shape[-1]
        num_classes = len(np.unique(data["y_train"]))
        sample_size = 1000

        sample_indices = np.random.choice(
            data["X_train"].shape[0], sample_size, replace=False
        )
        X_train_sampled = data["X_train"][sample_indices, :]
        y_train_sampled = (
            torch.Tensor(data["y_train"][sample_indices]).unsqueeze(1).to(device)
        )
        X_train_sampled, X = (
            pad_x(torch.Tensor(X_train_sampled).unsqueeze(1)).to(device),
            pad_x(torch.Tensor(X).unsqueeze(1)).to(device),
        )

        logits = model(
            x_src=torch.cat([X_train_sampled, X], dim=0),
            y_src=torch.cat(
                [y_train_sampled, torch.zeros((len(X), 1)).to(device)], dim=0
            ),
            eval_pos=len(X_train_sampled),
            normalization=True,
            outlier_clipping=False,
            nan_replacement=False,
            used_features=num_features,
        )
        embedding = logits.squeeze(1)[:, :num_classes].cpu().numpy()
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding
    elif embedding_type == "layer1":
        X_padded = pad_x(torch.Tensor(X).unsqueeze(1))
        if embedding_layer:
            embedding = embedding_layer(X_padded).squeeze(1).cpu().numpy()
        else:
            embedding = model.encoder(X_padded).squeeze(1).cpu().numpy()
        if y is not None:
            y_embedding = model.y_encoder(torch.Tensor(y).unsqueeze(1).to(device))
            embedding += y_embedding.squeeze(1).cpu().numpy()
        # embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding
    elif embedding_type == "layer2":
        X_padded = pad_x(torch.Tensor(X).unsqueeze(1)).to(device)
        embedding = model.encoder(X_padded)
        w_q, w_k, w_v = model.transformer_encoder[0].self_attn.in_proj_weight.chunk(3)
        b_q, b_k, b_v = model.transformer_encoder[0].self_attn.in_proj_bias.chunk(3)

        # TODO need to be fixed
        if not is_context:
            embedding = torch.nn.functional.linear(embedding, w_q, b_q)
            # embedding = torch.stack(embedding.chunk(4, -1), -1)
        else:
            embedding = torch.nn.functional.linear(embedding, w_k, b_k)
            # embedding = torch.stack(embedding.chunk(4, -1), -1)
        embedding = embedding.squeeze(1).cpu().numpy()
        # embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding
    else:
        raise Exception("Error: Invalid embedding type...")


import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm import tqdm

class PFKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, args, embedding_type='raw', device='cuda:0', knn_mode='multiclass', 
                 use_one_hot_emb=False, fix_missing=False, normalize_data=True):
        self.model = model
        self.args = args
        self.embedding_type = embedding_type
        self.device = device
        self.knn_mode = knn_mode
        self.use_one_hot_emb = use_one_hot_emb
        self.fix_missing = fix_missing
        self.normalize_data = normalize_data

    def fit(self, X, y):
        print("test")
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        self.X_train_ = X
        self.y_train_ = y
        
        # Create one-hot encoded version of X if needed
        if self.use_one_hot_emb:
            self.X_train_one_hot_ = self._one_hot_encode(X)
        
        # Initialize the FAISS index
        if self.knn_mode == 'multiclass':
            self.faiss_knn_ = MulticlassFaiss(
                self._compute_embedding(self.X_train_one_hot_ if self.use_one_hot_emb else self.X_train_),
                self.X_train_,
                self.y_train_
            )
        else:
            self.faiss_knn_ = SingleclassFaiss(
                self._compute_embedding(self.X_train_one_hot_ if self.use_one_hot_emb else self.X_train_),
                self.y_train_
            )
        
        print(self.args)
        self.sizes_per_class_ = get_sizes_per_class(self.args, self.y_train_, self.n_classes_)
        print("sizes_per_class_", self.sizes_per_class_)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        y_pred = []

        print("X.shape", X.shape)
        
        for x in tqdm(X):
            x = x.reshape(1, -1)  # Reshape to 2D array
            if self.knn_mode == 'multiclass':
                indices_X_nni, y_nni = self.faiss_knn_.get_knn_indices(
                    self._compute_embedding(self._one_hot_encode(x) if self.use_one_hot_emb else x),
                    self.sizes_per_class_
                )
                X_nni = np.concatenate(
                    [self.X_train_[self.y_train_ == i][indices] for i, indices in enumerate(indices_X_nni)],
                    axis=1
                )
                X_nni = np.swapaxes(X_nni, 0, 1)
                y_nni = y_nni.reshape(-1, 1).repeat(1, axis=1)
            else:
                indices_X_nni, y_nni = self.faiss_knn_.get_knn_indices(
                    self._compute_embedding(self._one_hot_encode(x) if self.use_one_hot_emb else x),
                    sum(self.sizes_per_class_)
                )
                X_nni = self.X_train_[indices_X_nni]
                X_nni = np.swapaxes(X_nni, 0, 1)
                y_nni = np.swapaxes(y_nni, 0, 1)

            X_nni, y_nni = (
                self._pad_x(torch.Tensor(X_nni)).to(self.device),
                torch.Tensor(y_nni).to(self.device),
            )
            x = self._pad_x(torch.Tensor(x).unsqueeze(0)).to(self.device)

            if self.fix_missing:
                X_nni, y_nni = self._fix_missing_data(X_nni, y_nni, indices_X_nni)

            logits = self.model(
                x_src=torch.cat([X_nni, x], dim=0),
                y_src=torch.cat([y_nni, torch.zeros((1, 1)).to(self.device)], dim=0),
                eval_pos=len(X_nni),
                normalization=self.normalize_data,
                outlier_clipping=False,
                nan_replacement=False,
                used_features=x.shape[-1],
            )

            logits = logits.squeeze(0)[:, :self.n_classes_]
            y_pred.append(logits.argmax().item())

        return np.array(y_pred)

    def _compute_embedding(self, X):
        return compute_embedding(
            self.model,
            {"X_train": self.X_train_, "y_train": self.y_train_},
            X,
            embedding_type=self.embedding_type,
            device=self.device
        )

    def _one_hot_encode(self, X):
        # Implement one-hot encoding logic here
        pass

    def _pad_x(self, x):
        return pad_x(x)

    def _fix_missing_data(self, X_nni, y_nni, indices_X_nni):
        # Implement fix_missing logic here
        return X_nni, y_nni
