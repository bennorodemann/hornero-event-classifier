"""Threshold-based classifier implementation."""

from typing import Iterable, Sequence, Self

import numpy as np
from numpy.typing import NDArray

from hornero_event_classifier.classifiers.base import Classifier
from hornero_event_classifier.classifiers.metrics import Metric


class ThresholdClassifier(Classifier):
    """A constant weighted threshold classifier over metric vectors.

    :param metrics: Metrics used to compute the score.
    :type metrics: Iterable[Metric]
    :param weights: Per-metric weights matching ``metrics`` order.
    :type weights: Sequence[float]
    :param threshold: Classification threshold, defaults to 0.5.
    :type threshold: float, optional
    """

    def __init__(
        self,
        metrics: Iterable[Metric],
        weights: Sequence[float],
        threshold: float = 0.5,
    ) -> None:
        metrics = tuple(metrics)
        if len(metrics) != len(weights):
            raise ValueError("metrics and weights must have same length")
        self.weights: list[float] = list(weights)
        self.threshold: float = threshold
        super().__init__(metrics)

    @classmethod
    def from_dict(cls, metric_weights: dict[Metric, float], threshold: float = 0.5) -> Self:
        """Create a classifier from a metric-to-weight mapping."""
        metrics, weights = zip(*metric_weights.items())
        return cls(metrics, weights, threshold)

    def classify_matrix(self, matrix: NDArray[np.floating]) -> NDArray[np.bool]:
        """Classify rows of the metric matrix using a weighted sum."""
        return (matrix * self.weights).sum(axis=1) > self.threshold

    def clean_seq(self, segments: tuple[Sequence, ...], raw_classifications: NDArray[np.bool]) -> NDArray[np.bool]:
        """Return the raw classifications unchanged."""
        return raw_classifications
