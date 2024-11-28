from sklearn.cluster import KMeans, HDBSCAN, DBSCAN
from tabpfn import TabPFNClassifier
import numpy as np
import torch

class KmeanTabPFNClassifier:
    def __init__(self,n_clusters=3, base_classifier=None):
        if base_classifier is None:
            base_classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=1)
        self.base_classifier = base_classifier
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)

    def fit(self, X, y, overwrite_warning=False):
        self.overwrite_warning = overwrite_warning
        # Cluster the data
        self.kmeans.fit(X.detach().cpu().numpy())
        self.X_train = X
        self.y_train = y
        
        # # Train a classifier for each cluster
        # for cluster in range(self.n_clusters):
        #     cluster_indices = np.where(clusters == cluster)
        #     X_cluster = X[cluster_indices]
        #     y_cluster = y[cluster_indices]
        #     self.classifiers[cluster].fit(X_cluster, y_cluster)

    def predict(self, X):
        clusters = self.kmeans.predict(X.detach().cpu().numpy())
        
        # Predict using the classifier for the corresponding cluster
        predictions = np.zeros(X.shape[0])
        for cluster in range(self.n_clusters):
            train_clusters = self.kmeans.predict(self.X_train.detach().cpu().numpy())
            train_cluster_indices = np.where(train_clusters == cluster)
            X_train_cluster = self.X_train[train_cluster_indices]
            y_train_cluster = self.y_train[train_cluster_indices]
            print(f"Shape of X_train_cluster: {X_train_cluster.shape}")
            print(f"Number of each classes", np.unique(y_train_cluster, return_counts=True))
            self.base_classifier.fit(X_train_cluster, y_train_cluster, overwrite_warning=self.overwrite_warning)
            cluster_indices = np.where(clusters == cluster)
            if len(cluster_indices[0]) > 0:
                predictions[cluster_indices] = self.base_classifier.predict(X[cluster_indices])
        
        return predictions
    
    def predict_proba(self, X):
        clusters = self.kmeans.predict(X.detach().cpu().numpy())
        
        # Predict probabilities using the classifier for the corresponding cluster
        proba_predictions = torch.zeros((X.shape[0], len(np.unique(self.y_train)))).to(self.base_classifier.device)
        for cluster in range(self.n_clusters):
            train_clusters = self.kmeans.predict(self.X_train.detach().cpu().numpy())
            train_cluster_indices = np.where(train_clusters == cluster)
            X_train_cluster = self.X_train[train_cluster_indices]
            y_train_cluster = self.y_train[train_cluster_indices]
            print(f"Shape of X_train_cluster: {X_train_cluster.shape}")
            print(f"Number of each classes", np.unique(y_train_cluster, return_counts=True))
            self.base_classifier.fit(X_train_cluster, y_train_cluster, overwrite_warning=self.overwrite_warning)
            cluster_indices = np.where(clusters == cluster)
            print(X, cluster_indices)
            print(X[cluster_indices].shape)
            if len(cluster_indices[0]) > 0:
                proba_predictions[cluster_indices] = self.base_classifier.predict_proba(X[cluster_indices].squeeze(0))
        
        return proba_predictions
    

class HDBSCANTabPFNClassifier:
    def __init__(self, base_classifier=None, min_cluster_size=100):
        if base_classifier is None:
            base_classifier = TabPFNClassifier(device="cpu", N_ensemble_configurations=1)
        self.base_classifier = base_classifier
        #self.n_clusters = n_clusters
        #self.hdbscan = HDBSCAN(min_cluster_size=min_cluster_size)
        self.hdbscan = DBSCAN()

    def fit(self, X, y, overwrite_warning=False):
        self.overwrite_warning = overwrite_warning
        self.X_train = X
        self.y_train = y
        
        # # Train a classifier for each cluster
        # for cluster in range(self.n_clusters):
        #     cluster_indices = np.where(clusters == cluster)
        #     X_cluster = X[cluster_indices]
        #     y_cluster = y[cluster_indices]
        #     self.classifiers[cluster].fit(X_cluster, y_cluster)

    def predict(self, X):
        X_combined = torch.cat((self.X_train, X), dim=0).detach().cpu().numpy()
        self.hdbscan.fit(X_combined)
        train_clusters = self.hdbscan.labels_[:self.X_train.shape[0]]
        test_clusters = self.hdbscan.labels_[self.X_train.shape[0]:]
        
        # Predict using the classifier for the corresponding cluster
        predictions = np.zeros(X.shape[0])
        print(self.hdbscan.labels_)
        print(train_clusters)
        for cluster in range(-1, self.hdbscan.labels_.max()):
            train_cluster_indices = np.where(train_clusters == cluster)
            X_train_cluster = self.X_train[train_cluster_indices]
            y_train_cluster = self.y_train[train_cluster_indices]
            print(f"Shape of X_train_cluster: {X_train_cluster.shape}")
            print(f"Number of each classes", np.unique(y_train_cluster, return_counts=True))
            if len(y_train_cluster) <= 0:
                continue
            self.base_classifier.fit(X_train_cluster, y_train_cluster, overwrite_warning=self.overwrite_warning)
            cluster_indices = np.where(test_clusters == cluster)
            if len(cluster_indices[0]) > 0:
                predictions[cluster_indices] = self.base_classifier.predict(X[cluster_indices])
        
        return predictions
    
    def predict_proba(self, X):
        X_combined = torch.cat((self.X_train, X), dim=0).detach().cpu().numpy()
        self.hdbscan.fit(X_combined)
        train_clusters = self.hdbscan.labels_[:self.X_train.shape[0]]
        test_clusters = self.hdbscan.labels_[self.X_train.shape[0]:]
        
        # Predict probabilities using the classifier for the corresponding cluster
        proba_predictions = torch.zeros((X.shape[0], len(np.unique(self.y_train)))).to(self.base_classifier.device)
        for cluster in range(self.hdbscan.labels_.max()):
            train_cluster_indices = np.where(train_clusters == cluster)
            X_train_cluster = self.X_train[train_cluster_indices]
            y_train_cluster = self.y_train[train_cluster_indices]
            print(f"Shape of X_train_cluster: {X_train_cluster.shape}")
            print(f"Number of each classes", np.unique(y_train_cluster, return_counts=True))
            self.base_classifier.fit(X_train_cluster, y_train_cluster, overwrite_warning=self.overwrite_warning)
            cluster_indices = np.where(test_clusters == cluster)
            print(X, cluster_indices)
            print(X[cluster_indices].shape)
            if len(cluster_indices[0]) > 0:
                proba_predictions[cluster_indices] = self.base_classifier.predict_proba(X[cluster_indices].squeeze(0))
        
        return proba_predictions

