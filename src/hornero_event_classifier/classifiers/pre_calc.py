from __future__ import annotations

from math import atan2, pi
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

if TYPE_CHECKING:
    from hornero_event_classifier.core.data import BBox

_index_counter = 0


class Dependency(Protocol):
    order: int
    dependencies: Sequence[Dependency]

    def __call__(self, boxes: Sequence[BBox]) -> Any: ...


def _init_dependency(dependencies: Sequence[Dependency] = ()):
    def _order_ring_counter(func: Callable[[Sequence[BBox]], Any]) -> Dependency:
        global _index_counter
        func.order = _index_counter
        func.dependencies = dependencies
        _index_counter += 1
        return func  # type: ignore

    return _order_ring_counter


@_init_dependency()
def global_rings(boxes: Sequence[BBox]):
    for box in boxes:
        box.metrics_cache[global_rings] = tuple(box.frame_obj.rings)


@_init_dependency()
def local_rings(boxes: Sequence[BBox]):
    if global_rings in boxes[0].metrics_cache:
        for box in boxes:
            box.metrics_cache[local_rings] = tuple(ring for ring in box.metrics_cache[global_rings] if ring.within(box))
    else:
        for box in boxes:
            box.metrics_cache[local_rings] = tuple(ring for ring in box.frame_obj.rings if ring.within(box))


@_init_dependency((local_rings,))
def local_ring_counts(boxes: Sequence[BBox]):
    for box in boxes:
        box.metrics_cache[local_ring_counts] = len(box.metrics_cache[local_rings])


def _score_ring(bird: BBox, ring: BBox):
    y_score = (ring.y - bird.ymin) / bird.height
    x_score = 1 - (abs(ring.x - bird.x) / (bird.width / 2))
    r_score = abs(atan2(ring.x - bird.x, -(ring.y - bird.y))) / pi
    w = 1 / 3
    return (w * y_score) + (w * x_score) + (w * r_score)


@_init_dependency((local_rings,))
def local_y_score(boxes: Sequence[BBox]):
    for box in boxes:
        box.metrics_cache[local_y_score] = tuple(
            (ring.y - box.ymin) / box.height for ring in box.metrics_cache[local_rings]
        )


@_init_dependency((local_rings,))
def local_x_score(boxes: Sequence[BBox]):
    for box in boxes:
        box.metrics_cache[local_x_score] = tuple(
            1 - (abs(ring.x - box.x) / (box.width / 2)) for ring in box.metrics_cache[local_rings]
        )


@_init_dependency((local_rings,))
def local_rad_score(boxes: Sequence[BBox]):
    for box in boxes:
        box.metrics_cache[local_rad_score] = tuple(
            abs(atan2(ring.x - box.x, -(ring.y - box.y))) / pi for ring in box.metrics_cache[local_rings]
        )


@_init_dependency((local_y_score, local_x_score, local_rad_score))
def local_ring_scores(boxes: Sequence[BBox]):
    for box in boxes:
        box.metrics_cache[local_ring_scores] = tuple(
            sum(scores) / 3
            for scores in zip(
                box.metrics_cache[local_y_score], box.metrics_cache[local_x_score], box.metrics_cache[local_rad_score]
            )
        )


@_init_dependency((local_rings,))
def local_real_rings(boxes: Sequence[BBox]):
    for box in boxes:
        box.metrics_cache[local_real_rings] = tuple(ring for ring in box.metrics_cache[local_rings] if ring.real)


@_init_dependency((local_rings,))
def local_ring_rotations(boxes: Sequence[BBox]):
    for box in boxes:
        box.metrics_cache[local_ring_rotations] = tuple(
            atan2(ring.x - box.x, ring.y - box.y) / pi for ring in box.metrics_cache[local_rings]
        )


@_init_dependency((local_rings,))
def local_ring_x_pos(boxes: Sequence[BBox]):
    for box in boxes:
        box.metrics_cache[local_ring_x_pos] = tuple(
            (ring.x - box.xmin) / box.width for ring in box.metrics_cache[local_rings]
        )


@_init_dependency((local_rings,))
def local_ring_global_x_pos(boxes: Sequence[BBox]):
    for box in boxes:
        box.metrics_cache[local_ring_global_x_pos] = tuple(
            ring.x / ring.frame_obj.width for ring in box.metrics_cache[local_rings]
        )


# @_init_dependency((global_rings,))
# def global_real_rings(boxes: Sequence[BBox]):
#     for box in boxes:
#         box.metrics_cache[global_real_rings] = tuple(ring for ring in box.metrics_cache[global_rings] if ring.real)


# @_init_dependency((global_rings,))
# def nearest_ring(boxes: Sequence[BBox]):
#     for box in boxes:
#         rings = box.metrics_cache[global_rings]
#         if len(rings):
#             box.metrics_cache[nearest_ring] = min(rings, key=box.distance_to)
#         else:
#             box.metrics_cache[nearest_ring] = None


# @_init_dependency((local_rings,))
# def local_real_rings(boxes: Sequence[BBox]):
#     for box in boxes:
#         box.metrics_cache[local_real_rings] = tuple(ring for ring in box.metrics_cache[local_rings] if ring.real)


# @_init_dependency((local_rings,))
# def avg_rel_ring_pos(boxes: Sequence[BBox]):
#     for box in boxes:
#         rings = box.metrics_cache[local_rings]
#         n_rings = len(rings)
#         if n_rings:
#             rx = sum(ring.x for ring in rings) / n_rings
#             ry = sum(ring.y for ring in rings) / n_rings
#             box.metrics_cache[avg_rel_ring_pos] = (rx, ry)
#         else:
#             box.metrics_cache[avg_rel_ring_pos] = (float("nan"), float("nan"))


# @_init_dependency([global_rings])
# def global_has_rings(boxes: Sequence[BBox]):
#     for box in boxes:
#         box.metrics_cache[global_has_rings] = len(box.metrics_cache[global_rings]) > 0


# @_init_dependency([local_rings])
# def local_has_rings(boxes: Sequence[BBox]):
#     for box in boxes:
#         box.metrics_cache[local_has_rings] = len(box.metrics_cache[local_rings]) > 0
