"""Dependency builders for classifier metrics.

.. currentmodule:: hornero_event_classifier.core
"""

from __future__ import annotations

from math import atan2, pi
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, Iterable
from itertools import count

if TYPE_CHECKING:
    from hornero_event_classifier.core.data import BBox, ItemType

_index_counter: count = count()


class Dependency(Protocol):  # pylint: disable=too-few-public-methods
    """A Protocol describing a callable with an execution order and optional dependencies.

    The callable should accept a target :py:class:`~enum.ItemType` and a :py:type:`Sequence` of
    :py:attr:`~enums.ItemType.BIRD` :py:class:`~data.BBox`\\s and add results to :py:attr:`data.BBox.metrics_cache`
    using itself as the key.
    """

    #: the order index of the :py:class:`Dependency`
    order: int
    #: other :py:class:`Dependency`\s that this :py:class:`Dependency` relies on
    dependencies: Sequence[Dependency]

    def __call__(self, targets: tuple[ItemType, ...], boxes: Sequence[BBox]) -> Any: ...


def _init_dependency(dependencies: Sequence[Dependency] = ()):
    def _order_ring_counter(func: Callable[[tuple[ItemType, ...], Sequence[BBox]], Any]) -> Dependency:
        func.order = next(_index_counter)
        func.dependencies = dependencies
        return func  # type: ignore

    return _order_ring_counter


@_init_dependency()
def global_items(targets: tuple[ItemType, ...], boxes: Sequence[BBox]):
    """Cache a :py:type:`tuple` of all ``target`` :py:class:`~data.Item`\\s in the current frame.

    :param targets: target :py:class:`~enums.ItemType`\\s to compare with ``boxes``
    :type targets: tuple[ItemType, ...]
    :param boxes: :py:attr:`~enums.ItemType.BIRD` :py:class:`~data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    for box in boxes:
        # box.metrics_cache[global_rings] = tuple(box.frame_obj.rings)
        box.metrics_cache[targets][global_items] = tuple(box.frame_obj.get_items(*targets))


@_init_dependency()
def local_items(targets: tuple[ItemType, ...], boxes: Sequence[BBox]):
    """Cache ``target`` item whose centers fall within each bird :py:class:`~data.BBox`.

    :param targets: target :py:class:`~enums.ItemType`\\s to compare with ``boxes``
    :type targets: tuple[ItemType, ...]
    :param boxes: :py:attr:`~enums.ItemType.BIRD` :py:class:`~data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    # If global rings are already cached, reuse them; otherwise compute them from the current frame.
    if global_items in boxes[0].metrics_cache[targets]:
        for box in boxes:
            box.metrics_cache[targets][local_items] = tuple(
                item for item in box.metrics_cache[targets][global_items] if item.within(box)
            )
    else:
        for box in boxes:
            box.metrics_cache[targets][local_items] = tuple(
                item for item in box.frame_obj.get_items(*targets) if item.within(box)
            )


@_init_dependency((local_items,))
def local_item_counts(targets: tuple[ItemType, ...], boxes: Sequence[BBox]):
    """Cache the number of :py:func:`local_items` per bird. (depends on :py:func:`local_items`)

    :param targets: target :py:class:`~enums.ItemType`\\s to compare with ``boxes``
    :type targets: tuple[ItemType, ...]
    :param boxes: :py:attr:`~enums.ItemType.BIRD` :py:class:`~data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    for box in boxes:
        box.metrics_cache[targets][local_item_counts] = len(box.metrics_cache[targets][local_items])


@_init_dependency((local_items,))
def local_y_score(targets: tuple[ItemType, ...], boxes: Sequence[BBox]):
    """Cache proportional y positions of :py:func:`local_items` within each bird :py:class:`BBox`.

    0 = top of bounding box, 1 = bottom of bounding box. (depends on :py:func:`local_items`)

    :param targets: target :py:class:`~enums.ItemType`\\s to compare with ``boxes``
    :type targets: tuple[ItemType, ...]
    :param boxes: :py:attr:`~enums.ItemType.BIRD` :py:class:`~data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    # ymin: is the top of the bounding box
    for box in boxes:
        box.metrics_cache[targets][local_y_score] = tuple(
            (item.y - box.ymin) / box.height for item in box.metrics_cache[targets][local_items]
        )


@_init_dependency((local_items,))
def local_item_x_pos(targets: tuple[ItemType, ...], boxes: Sequence[BBox]):
    """Cache proportional x positions of :py:func:`local_items` within each bird :py:class:`BBox`.

    0 = leftmost side, 1 = rightmost side of bounding box. (depends on :py:func:`local_items`)

    :param targets: target :py:class:`~enums.ItemType`\\s to compare with ``boxes``
    :type targets: tuple[ItemType, ...]
    :param boxes: :py:attr:`~enums.ItemType.BIRD` :py:class:`~data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    for box in boxes:
        box.metrics_cache[targets][local_item_x_pos] = tuple(
            (item.x - box.xmin) / box.width for item in box.metrics_cache[targets][local_items]
        )


@_init_dependency((local_items,))
def local_item_rotations(targets: tuple[ItemType, ...], boxes: Sequence[BBox]):
    """Cache proportional angles from the bird center to each :py:func:`local_items`.

    1 and -1 = straight up (90°), 0 = straight down (-90°). (depends on :py:func:`local_items`)

    :param targets: target :py:class:`~enums.ItemType`\\s to compare with ``boxes``
    :type targets: tuple[ItemType, ...]
    :param boxes: :py:attr:`~enums.ItemType.BIRD` :py:class:`~data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    for box in boxes:
        box.metrics_cache[targets][local_item_rotations] = tuple(
            atan2(item.x - box.x, item.y - box.y) / pi for item in box.metrics_cache[targets][local_items]
        )


@_init_dependency((local_items,))
def local_real_items(targets: tuple[ItemType, ...], boxes: Sequence[BBox]):
    """Cache the :py:func:`local_items` that are real (non-interpolated) :py:class:`BBox`\\s.

    :param targets: target :py:class:`~enums.ItemType`\\s to compare with ``boxes``
    :type targets: tuple[ItemType, ...]
    :param boxes: :py:attr:`~enums.ItemType.BIRD` :py:class:`~data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    for box in boxes:
        box.metrics_cache[targets][local_real_items] = tuple(
            item for item in box.metrics_cache[targets][local_items] if item.real
        )
