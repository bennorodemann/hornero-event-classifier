from typing import Callable

from hornero_event_classifier.core.data import BBox

FilterFunc = Callable[[BBox, BBox], bool]


def make_gap_filter(gap: int) -> FilterFunc:
    def gap_filter(box1: BBox, box2: BBox) -> bool:
        return (box2.frame - box1.frame) >= gap

    return gap_filter


def frame_touch_filter(box1: BBox, box2: BBox) -> bool:
    frame = box1.frame_obj
    return box1.touching_boundary(frame.width, frame.height, 5) or box2.touching_boundary(frame.width, frame.height, 5)


def make_buffer_filter(buffer: int) -> FilterFunc:
    def buffer_filter(box1: BBox, _: BBox) -> bool:
        return box1.item_obj.start + buffer < box1.frame < box1.item_obj.end - buffer

    return buffer_filter


def boundary_filter(box1: BBox, _: BBox) -> bool:
    return any(
        box1.frame in (bird.item_obj.start, bird.item_obj.end) for bird in box1.frame_obj.birds if bird is not box1.item_obj
    )


def invert_filter(func: FilterFunc):
    def inverted_filter(box1: BBox, box2: BBox):
        return not func(box1, box2)

    return inverted_filter
