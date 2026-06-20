"""Metric definitions and their registered logic.

These metrics are the primary inputs to internal classifiers.

.. currentmodule:: hornero_event_classifier.classifiers
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Callable,
    Sequence,
)

import numpy as np
from numpy.typing import NDArray

import hornero_event_classifier.classifiers.dependencies as req

# from hornero_event_classifier.core import ItemType

if TYPE_CHECKING:
    from hornero_event_classifier.core.data import BBox


class Metric(IntEnum):
    """An Enum class of metric options that can be passed to :py:class:`base.Classifier`\\s.

    All metrics return values between 0 and 1.
    """

    ITEM_PRESENCE = auto()
    """Proportion of frames that contain any :py:func:`dependencies.local_items`.

    (0 = no frames contained items, 1 = all frames contained a item)
    """
    CENTER_ITEM_PRESENCE = auto()
    """The same as :py:attr:`Metric.ITEM_PRESENCE` but ignoring the first and last quarter of all frames.
    
    (0 = no frames contained items, 1 = all frames contained a item)
    """
    PER_OWNERSHIP = auto()
    """The proportion of :py:func:`dependencies.local_items` / :py:func:`dependencies.global_items` per frame.
    
    (0 = no items in frame are local, 1 = all item in frame are local, NaN = no global items)"""
    ITEM_COUNT = auto()
    """A proportional representation of the number of :py:func:`dependencies.local_items` in each frame.

    Capped at 5 items. (NaN = no local items, 1 = 5 or more local items present)
    """
    AVG_ITEM_CONF = auto()
    """The average confidence value of all :py:func:`dependencies.local_items` per frame.
    
    (0 = not confident, 1 = confident, NaN = no local items)
    """
    AVG_ITEM_REAL = auto()
    """The percent of real (non-interpolated) :py:func:`dependencies.local_items` per frame.
    
    (0 = no real local items, 1 = all real local items, NaN = no local items)
    """
    AVG_X_SCORE = auto()
    """The average proportional x distance of :py:func:`dependencies.local_items` to the center of the bird's
    :py:class:`~hornero_event_classifier.core.data.BBox` per frame. 
    
    (0 = in the center, 1 = on left or right side of bbox, NaN = no local items)
    """
    AVG_Y_SCORE = auto()
    """The average proportional y position of :py:func:`dependencies.local_items` in the bird's
    :py:class:`~hornero_event_classifier.core.data.BBox` per frame. 
    
    (0 = top, 1 = bottom of bbox, NaN = no local items)
    """
    AVG_RAD_SCORE = auto()
    """The average proportional angle of :py:func:`dependencies.local_items` to the center of the bird's
    :py:class:`~hornero_event_classifier.core.data.BBox` per frame. 
    
    (0 = straight up (90°), 1 = straight down (-90°), NaN = no local items)
    """
    X_STD = auto()
    """The standard deviation of :py:func:`dependencies.local_items` relative x position in the bird's
    :py:class:`~hornero_event_classifier.core.data.BBox` across all frames.
    
    (0 = no deviation, 1 = full deviation, NaN = no local items)
    """
    RAD_STD = auto()
    """The standard deviation of :py:func:`dependencies.local_items` relative angle to the center of the bird's
    :py:class:`~hornero_event_classifier.core.data.BBox` across all frames.
    
    (0 = no deviation, 1 = full deviation, NaN = no local items)
    """
    GLOBAL_X_STD = auto()
    """The standard deviation of :py:func:`dependencies.local_items` relative x position in relation to the entire frame
    across all frames.
    
    (0 = no deviation, 1 = full deviation, NaN = no local items)
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __str__(self) -> str:
        return self.name


class MetricRegistry[O]:
    """A registry class for :py:class:`Metric` logic functions."""

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
        """Register a new :py:class:`Metric` logic function. (Designed to be used as a decorator)

        :param key: corresponding :py:class:`Metric`
        :type key: Metric
        :param takes: :py:class:`dependencies.Dependency`\\s that will be passed as input
            arguments, defaults to ()
        :type takes: Sequence[req.Dependency] | req.Dependency, optional
        :return: a function that accepts the actual logic function and registers it the the provided :py:class:`Metric`
        :rtype: Callable[[Callable[..., O]], Callable[..., O]]
        """
        if not isinstance(takes, Sequence):
            takes = (takes,)

        def func(register_func: Callable[..., O]) -> Callable[..., O]:
            self._registry[key] = register_func
            self._takes[key] = takes
            return register_func

        return func

    def get(self, key: Metric) -> Callable[..., O]:
        """Get the corresponding logic function

        :param key: the search :py:class:`Metric`
        :type key: Metric
        :return: the corresponding logic function
        :rtype: Callable[..., O]
        """
        return self._registry[key]

    def get_args(self, key: Metric) -> Sequence[req.Dependency]:
        """Get the :py:class:`dependencies.Dependency` input arguments that the logic
        function requires.

        :param key: the search :py:class:`Metric`
        :type key: Metric
        :return: the :py:class:`dependencies.Dependency` the logic function takes as inputs
        :rtype: Sequence[req.Dependency]
        """
        return self._takes[key]

    def get_dependency_list(self, key: Metric) -> set[req.Dependency]:
        """Return all direct and nested dependencies required by a metric.

        :param key: Metric to inspect.
        :type key: Metric
        :return: Set of dependency functions required by the metric.
        :rtype: set[req.Dependency]
        """
        out = set(self._takes[key])
        for take in self._takes[key]:
            out |= self._get_dependencies(take)
        return out

    def _get_dependencies(self, dependency: req.Dependency) -> set[req.Dependency]:
        out = set(dependency.dependencies)
        for depend in dependency.dependencies:
            out |= self._get_dependencies(depend)
        return out


metric_func_registry: MetricRegistry[list[float] | list[np.floating] | NDArray[np.floating]] = MetricRegistry()


@metric_func_registry.register(Metric.ITEM_PRESENCE, (req.local_item_counts,))
def get_item_presence(_: list[BBox], has_items: list[int]) -> list[float]:
    return [float(v > 0) for v in has_items]


@metric_func_registry.register(Metric.CENTER_ITEM_PRESENCE, (req.local_item_counts,))
def get_center_item_presence(_: list[BBox], has_items: list[int]) -> NDArray[np.floating]:
    out: NDArray[np.float64] = np.array([float(v > 0) for v in has_items], np.float64)
    q_len = int(len(out) / 4)
    out[:q_len] = np.nan
    out[-q_len:] = np.nan
    return out


@metric_func_registry.register(Metric.PER_OWNERSHIP, (req.local_item_counts, req.global_items))
def get_per_ownership(_: list[BBox], local_item_counts: list[int], global_items: list[tuple[BBox]]) -> list[float]:
    return [loc / len(glob) if glob else float("nan") for loc, glob in zip(local_item_counts, global_items)]


@metric_func_registry.register(Metric.ITEM_COUNT, (req.local_item_counts))
def get_item_count(_: list[BBox], item_counts: list[int]) -> list[float]:
    return [min(5, item_count) / 5 if item_count else float("nan") for item_count in item_counts]


@metric_func_registry.register(Metric.AVG_ITEM_CONF, req.local_real_items)
def get_avg_item_conf(_: list[BBox], all_items: list[tuple[BBox, ...]]) -> list[float]:
    return [sum(item.conf for item in items) / len(items) if items else float("nan") for items in all_items]


@metric_func_registry.register(Metric.AVG_ITEM_REAL, req.local_items)
def get_avg_item_real(_: list[BBox], all_items: list[tuple[BBox, ...]]) -> list[float]:
    return [sum(item.real for item in items) / len(items) if items else float("nan") for items in all_items]


@metric_func_registry.register(Metric.AVG_X_SCORE, req.local_item_x_pos)
def get_avg_x_score(_: list[BBox], all_positions: list[tuple[float, ...]]) -> list[float]:
    return [
        (
            sum(2 * abs(item_pos - 0.5) for item_pos in item_positions) / len(item_positions)
            if item_positions
            else float("nan")
        )
        for item_positions in all_positions
    ]


@metric_func_registry.register(Metric.AVG_Y_SCORE, req.local_y_score)
def get_avg_y_score(_: list[BBox], all_scores: list[tuple[float, ...]]) -> list[float]:
    return [sum(item_scores) / len(item_scores) if item_scores else float("nan") for item_scores in all_scores]


@metric_func_registry.register(Metric.AVG_RAD_SCORE, req.local_item_rotations)
def get_avg_rad_score(_: list[BBox], all_scores: list[tuple[float, ...]]) -> list[float]:
    return [
        sum(1 - abs(item_rad) for item_rad in item_rads) / len(item_rads) if item_rads else float("nan")
        for item_rads in all_scores
    ]


@metric_func_registry.register(Metric.RAD_STD, req.local_item_rotations)
def get_rad_std(_: list[BBox], all_rotations: list[tuple[float, ...]]) -> list[np.floating]:
    val = np.std([r for frame in all_rotations for r in frame] or [np.nan])
    return [val for _ in range(len(all_rotations))]


@metric_func_registry.register(Metric.X_STD, req.local_item_x_pos)
def get_x_std(_: list[BBox], all_rotations: list[tuple[float, ...]]) -> list[np.floating]:
    val = np.std([r for frame in all_rotations for r in frame] or [np.nan])
    return [val for _ in range(len(all_rotations))]


@metric_func_registry.register(Metric.GLOBAL_X_STD, req.local_items)
def get_global_x_std(_: list[BBox], local_items: list[tuple[BBox, ...]]) -> list[float]:
    flattened: list[BBox] = [b for frame in local_items for b in frame]
    val: float
    if len(flattened):
        frame_width = flattened[0].frame_obj.width
        val = float(np.std([box.x / frame_width for box in flattened]))
    else:
        val = np.nan
    return [val for _ in range(len(local_items))]


assert all(metric in metric_func_registry._registry for metric in Metric)
