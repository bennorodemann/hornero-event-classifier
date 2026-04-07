"""K-means based classifier implementation.

Designed for lightweight, unsupervised baselines in internal experiments.
"""

from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from hornero_event_classifier.classifiers.base import Classifier, SegmentCollection, Sequence
from hornero_event_classifier.classifiers.metrics import Metric
from hornero_event_classifier.classifiers.utils import moving_average, small_run_filter, square_filter
from hornero_event_classifier.core.data import Item


class KMeanClassifier(Classifier):
    """Unsupervised k-means classifier for ring/no-ring separation."""

    def __init__(self, metrics: Iterable[Metric]) -> None:
        """Initialize a k-means classifier with the required metrics."""
        super().__init__(metrics)

    def train(self, data: SegmentCollection):
        """Fit the k-means model and derive feature weights from cluster separation.

        :param data: Segment collection used for training.
        :type data: SegmentCollection
        :return: ``None``.
        :rtype: None
        """
        self.scaler = StandardScaler()
        self.model = KMeans(2, n_init=200)
        training_matrix = data.data[[(segment.end - segment.start + 1) > 300 for segment in data.segments]]
        transformed_matrix = self.scaler.fit_transform(training_matrix)
        self.model.fit(transformed_matrix)
        feat_var = transformed_matrix.var(axis=0, ddof=1)
        center_var = self.model.cluster_centers_.var(axis=0, ddof=1)
        sep_ratio = center_var / (feat_var + 1e-9)
        weights = sep_ratio / sep_ratio.max()
        self.weights = (weights * 0.8) + 0.2
        self.model = KMeans(2, n_init=200)
        self.model.fit(transformed_matrix * self.weights)
        no_ring_id = int(self.model.predict(self.scaler.transform(np.array([[0 for _ in self.metrics]])) * self.weights)[0])
        self.classification_lookup = np.array([True, False] if no_ring_id else [False, True])
        p_weights = {m.name: round(w, 4) for m, w in zip(self.metrics, self.weights)}
        print(f"weights: {p_weights}")

    def classify_matrix(self, matrix: NDArray[np.floating]) -> NDArray[np.bool]:
        """Classify rows of the metric matrix.

        :param matrix: Metric data matrix (rows = segments, columns = metrics).
        :type matrix: NDArray[np.floating]
        :return: Boolean classifications per row.
        :rtype: NDArray[np.bool]
        """
        results = self.model.predict(self.scaler.transform(matrix) * self.weights)
        return self.classification_lookup[results]

    def clean_seq(self, segments: tuple[Sequence, ...], raw_classifications: NDArray[np.bool]) -> NDArray[np.bool]:
        """Return raw classifications unchanged (no sequence cleaning)."""
        return raw_classifications
