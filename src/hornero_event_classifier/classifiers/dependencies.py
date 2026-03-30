from __future__ import annotations

from math import atan2, pi
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence
from itertools import count

if TYPE_CHECKING:
    from hornero_event_classifier.core.data import BBox

_index_counter: count = count()


class Dependency(Protocol):  # pylint: disable=too-few-public-methods
    """A Protocol describing a callable that has a specific order and may require other :py:class:`Dependency`\\s. The callable
    should accept a :py:type:`Sequence` of :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`
    :py:class:`~hornero_event_classifier.core.data.BBox`\\s and add information to
    :py:attr:`~hornero_event_classifier.core.data.BBox.metrics_cache` using itself as the key."""

    #: the order index of the :py:class:`Dependency`
    order: int
    #: other :py:class:`Dependency`\\s that this :py:class:`Dependency` relies on
    dependencies: Sequence[Dependency]

    def __call__(self, boxes: Sequence[BBox]) -> Any: ...


def _init_dependency(dependencies: Sequence[Dependency] = ()):
    def _order_ring_counter(func: Callable[[Sequence[BBox]], Any]) -> Dependency:
        func.order = next(_index_counter)
        func.dependencies = dependencies
        return func  # type: ignore

    return _order_ring_counter


@_init_dependency()
def global_rings(boxes: Sequence[BBox]):
    """Adds a :py:type:`tuple` of all ring :py:class:`~hornero_event_classifier.core.data.Item`\\s in the current frame.

    :param boxes: :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`
        :py:class:`~hornero_event_classifier.core.data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    for box in boxes:
        box.metrics_cache[global_rings] = tuple(box.frame_obj.rings)


@_init_dependency()
def local_rings(boxes: Sequence[BBox]):
    """Adds a :py:type:`tuple` of all ring :py:class:`~hornero_event_classifier.core.data.Item`\\s in the current frame where the
    center of the rings :py:class:`~hornero_event_classifier.core.data.BBox` is within each
    :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD` :py:class:`~hornero_event_classifier.core.data.BBox`.

    :param boxes: :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`
        :py:class:`~hornero_event_classifier.core.data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    # if global rings are already added pull global ring list from there otherwise get global ring list for itself
    if global_rings in boxes[0].metrics_cache:
        for box in boxes:
            box.metrics_cache[local_rings] = tuple(ring for ring in box.metrics_cache[global_rings] if ring.within(box))
    else:
        for box in boxes:
            box.metrics_cache[local_rings] = tuple(ring for ring in box.frame_obj.rings if ring.within(box))


@_init_dependency((local_rings,))
def local_ring_counts(boxes: Sequence[BBox]):
    """Adds a :py:type:`int` of the number of local rings. (depends on :py:func:`local_rings`)

    :param boxes: :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`
        :py:class:`~hornero_event_classifier.core.data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    for box in boxes:
        box.metrics_cache[local_ring_counts] = len(box.metrics_cache[local_rings])


@_init_dependency((local_rings,))
def local_y_score(boxes: Sequence[BBox]):
    """Adds a :py:type:`tuple` of :py:type:`float`\\s for each :py:func:`local_rings` representing the proportional y position of
    each ring within the :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`\\s
    :py:class:`~hornero_event_classifier.core.data.BBox`. 0 = top of bounding box, 1 = bottom of bounding box. (depends on
    :py:func:`local_rings`)

    :param boxes: :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`
        :py:class:`~hornero_event_classifier.core.data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    # ymin: is the top of the bounding box
    for box in boxes:
        box.metrics_cache[local_y_score] = tuple((ring.y - box.ymin) / box.height for ring in box.metrics_cache[local_rings])


@_init_dependency((local_rings,))
def local_ring_x_pos(boxes: Sequence[BBox]):
    """Adds a :py:type:`tuple` of :py:type:`float`\\s for each :py:func:`local_rings` representing the proportional x position of
    each ring within the :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`\\s
    :py:class:`~hornero_event_classifier.core.data.BBox`. 0 = left most side, 1 = right most side of bounding box. (depends on
    :py:func:`local_rings`)

    :param boxes: :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`
        :py:class:`~hornero_event_classifier.core.data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    for box in boxes:
        box.metrics_cache[local_ring_x_pos] = tuple((ring.x - box.xmin) / box.width for ring in box.metrics_cache[local_rings])


@_init_dependency((local_rings,))
def local_ring_rotations(boxes: Sequence[BBox]):
    """Adds a :py:type:`tuple` of :py:type:`float`\\s for each :py:func:`local_rings` representing the proportional angle of
    each ring to the center of the :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`\\s
    :py:class:`~hornero_event_classifier.core.data.BBox`. 1 and -1 = straight up (90°), 0 = straight down (-90°). (depends on
    :py:func:`local_rings`)

    :param boxes: :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`
        :py:class:`~hornero_event_classifier.core.data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    for box in boxes:
        box.metrics_cache[local_ring_rotations] = tuple(
            atan2(ring.x - box.x, ring.y - box.y) / pi for ring in box.metrics_cache[local_rings]
        )


@_init_dependency((local_rings,))
def local_real_rings(boxes: Sequence[BBox]):
    """Adds a :py:type:`tuple` of all :py:func:`local_rings` :py:class:`~hornero_event_classifier.core.data.Item`\\s that are
    (non-interpolated :py:class:`~hornero_event_classifier.core.data.BBox`\\s). (depends on :py:func:`local_rings`)

    :param boxes: :py:attr:`~hornero_event_classifier.core.enums.ItemType.BIRD`
        :py:class:`~hornero_event_classifier.core.data.BBox`\\s to modify
    :type boxes: Sequence[BBox]
    """
    for box in boxes:
        box.metrics_cache[local_real_rings] = tuple(ring for ring in box.metrics_cache[local_rings] if ring.real)
