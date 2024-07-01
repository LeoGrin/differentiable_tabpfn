# stdlib
from typing import Any, Dict

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.neighbors import NearestNeighbors

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader


class NearestRealNeighborDistance(BasicMetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_sanity.NearestRealNeighborDistance
        :parts: 1

    Computes the <reduction>(distance) from the synthetic data to the closest neighbor in the real data
    """

    @staticmethod
    def name() -> str:
        return "nearest_real_neighbor_distance"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if len(X_gt.columns) != len(X_syn.columns):
            raise ValueError(f"Incompatible dataframe {X_gt.shape} and {X_syn.shape}")

        # Swap X_gt and X_syn in the helper function call
        dist = BasicMetricEvaluator._helper_nearest_neighbor(X_syn, X_gt)

        # Normalize distances
        dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-8)
        return {self._reduction: float(self.reduction()(dist))}
