from __future__ import annotations

from enum import Flag, auto
from math import atan2, isnan, pi, sqrt
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    ParamSpec,
    Sequence,
    TypeVar,
)

import hornero_event_classifier.classifiers.pre_calc as req
import numpy as np
from hornero_event_classifier.core import ItemType

if TYPE_CHECKING:
    from hornero_event_classifier.core.data import BBox


class Metric(Flag):
    RING_PRESENCE = auto()
    RING_COUNT = auto()
    AVG_RING_CONF = auto()
    AVG_RING_REAL = auto()
    AVG_RING_SCORE = auto()
    AVG_X_SCORE = auto()
    AVG_Y_SCORE = auto()
    AVG_RAD_SCORE = auto()
    AVG_PLASTIC = auto()
    X_STD = auto()
    RAD_STD = auto()
    GLOBAL_X_STD = auto()


ring_presence = 0

K = TypeVar("K")
T = ParamSpec("T")
O = TypeVar("O")


class MetricRegistry(Generic[O]):
    def __init__(self) -> None:
        self._registry: dict[Metric, Callable[..., O]] = {}
        self._takes: dict[Metric, Sequence[req.Dependency]] = {}

    def __getitem__(self, key):
        return self.get(key)

    def register(
        self,
        key: Metric,
        takes: Sequence[req.Dependency] | req.Dependency = (),
    ) -> Callable[[Callable[..., O]], Callable[..., O]]:
        if not isinstance(takes, Sequence):
            takes = (takes,)

        def func(register_func: Callable[..., O]) -> Callable[..., O]:
            self._registry[key] = register_func
            self._takes[key] = takes
            return register_func

        return func

    def get(self, key: Metric) -> Callable[..., O]:
        return self._registry[key]

    def get_args(self, key: Metric) -> Sequence[req.Dependency]:
        return self._takes[key]

    def get_dependency_list(self, key: Metric) -> set[req.Dependency]:
        out = set(self._takes[key])
        for take in self._takes[key]:
            out |= self._get_dependencies(take)
        return out

    def _get_dependencies(self, dependency: req.Dependency) -> set[req.Dependency]:
        out = set(dependency.dependencies)
        for depend in dependency.dependencies:
            out |= self._get_dependencies(depend)
        return out


metric_func_registry: MetricRegistry[list[float] | list[np.floating]] = MetricRegistry()


@metric_func_registry.register(Metric.RING_PRESENCE, (req.local_ring_counts,))
def get_ring_presence(_: list[BBox], has_rings: list[bool]) -> list[float]:
    return [float(v > 0) for v in has_rings]


@metric_func_registry.register(Metric.RING_COUNT, (req.local_ring_counts))
def get_ring_count(_: list[BBox], ring_counts: list[int]) -> list[float]:
    return [min(3, ring_count) / 3 if ring_count else float("nan") for ring_count in ring_counts]


@metric_func_registry.register(Metric.AVG_RING_CONF, req.local_real_rings)
def get_avg_ring_conf(_: list[BBox], all_rings: list[tuple[BBox, ...]]) -> list[float]:
    return [sum(ring.conf for ring in rings) / len(rings) if rings else float("nan") for rings in all_rings]


@metric_func_registry.register(Metric.AVG_RING_REAL, req.local_rings)
def get_avg_ring_real(_: list[BBox], all_rings: list[tuple[BBox, ...]]) -> list[float]:
    return [sum(ring.real for ring in rings) / len(rings) if rings else float("nan") for rings in all_rings]


@metric_func_registry.register(Metric.AVG_RING_SCORE, req.local_ring_scores)
def get_avg_ring_score(_: list[BBox], all_scores: list[tuple[float, ...]]) -> list[float]:
    return [sum(ring_scores) / len(ring_scores) if ring_scores else float("nan") for ring_scores in all_scores]


@metric_func_registry.register(Metric.AVG_X_SCORE, req.local_x_score)
def get_avg_x_score(_: list[BBox], all_scores: list[tuple[float, ...]]) -> list[float]:
    return [sum(ring_scores) / len(ring_scores) if ring_scores else float("nan") for ring_scores in all_scores]


@metric_func_registry.register(Metric.AVG_Y_SCORE, req.local_y_score)
def get_avg_y_score(_: list[BBox], all_scores: list[tuple[float, ...]]) -> list[float]:
    return [sum(ring_scores) / len(ring_scores) if ring_scores else float("nan") for ring_scores in all_scores]


@metric_func_registry.register(Metric.AVG_RAD_SCORE, req.local_rad_score)
def get_avg_rad_score(_: list[BBox], all_scores: list[tuple[float, ...]]) -> list[float]:
    return [sum(ring_scores) / len(ring_scores) if ring_scores else float("nan") for ring_scores in all_scores]


@metric_func_registry.register(Metric.AVG_PLASTIC, req.local_rings)
def get_avg_plastic(_: list[BBox], all_rings: list[tuple[BBox, ...]]) -> list[float]:
    return [
        (
            sum(ring.item_obj.type == ItemType.RING_PLASTIC for ring in frame_rings) / len(frame_rings)
            if frame_rings
            else float("nan")
        )
        for frame_rings in all_rings
    ]


@metric_func_registry.register(Metric.RAD_STD, req.local_ring_rotations)
def get_rad_std(_: list[BBox], all_rotations: list[tuple[float, ...]]) -> list[np.floating]:
    val = np.std([r for frame in all_rotations for r in frame])
    return [val for _ in range(len(all_rotations))]


@metric_func_registry.register(Metric.X_STD, req.local_ring_x_pos)
def get_x_std(_: list[BBox], all_rotations: list[tuple[float, ...]]) -> list[np.floating]:
    val = np.std([r for frame in all_rotations for r in frame])
    return [val for _ in range(len(all_rotations))]


@metric_func_registry.register(Metric.GLOBAL_X_STD, req.local_ring_global_x_pos)
def get_global_x_std(_: list[BBox], all_rotations: list[tuple[float, ...]]) -> list[np.floating]:
    val = np.std([r for frame in all_rotations for r in frame])
    return [val for _ in range(len(all_rotations))]


assert all(metric in metric_func_registry._registry for metric in Metric)
