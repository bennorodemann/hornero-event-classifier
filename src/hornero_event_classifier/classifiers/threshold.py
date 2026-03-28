from typing import Iterable, Sequence, Self

import numpy as np
from numpy.typing import NDArray

from hornero_event_classifier.classifiers.base import Sequence, Classifier
from hornero_event_classifier.classifiers.metrics import Metric


class ThresholdClassifier(Classifier):
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
        metrics, weights = zip(*metric_weights.items())
        return cls(metrics, weights, threshold)

    def classify_matrix(self, matrix: NDArray[np.floating]) -> NDArray[np.bool]:
        return (matrix * self.weights).sum(axis=1) > self.threshold

    def clean_seq(self, segments: tuple[Sequence, ...], raw_classifications: NDArray[np.bool]) -> NDArray[np.bool]:
        return raw_classifications
