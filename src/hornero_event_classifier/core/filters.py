"""Filter function helpers for :py:class:`BBox` comparisons."""

from typing import Callable

from hornero_event_classifier.core.data import BBox

#: Filter type description
type FilterFunc = Callable[[BBox, BBox], bool]


def make_gap_filter(gap: int) -> FilterFunc:
    """Create a filter that passes when the frame gap between boxes is at least ``gap``.

    :param gap: Minimum number of frames between two boxes.
    :type gap: int
    :return: A filter function implementing the gap check.
    :rtype: FilterFunc
    """

    def gap_filter(box1: BBox, box2: BBox) -> bool:
        return (box2.frame - box1.frame) >= gap

    return gap_filter


def frame_touch_filter(box1: BBox, box2: BBox) -> bool:
    """Check whether either box touches the frame boundary.

    :param box1: First bounding box.
    :type box1: BBox
    :param box2: Second bounding box.
    :type box2: BBox
    :return: ``True`` if either box touches the frame boundary, otherwise ``False``.
    :rtype: bool
    """
    frame = box1.frame_obj
    return box1.touching_boundary(frame.width, frame.height, 5) or box2.touching_boundary(frame.width, frame.height, 5)


def make_buffer_filter(buffer: int) -> FilterFunc:
    """Create a filter that excludes boxes near the start or end of an item.

    :param buffer: Number of frames to exclude on both ends of an item's range.
    :type buffer: int
    :return: A filter function implementing the buffer check.
    :rtype: FilterFunc
    """

    def buffer_filter(box1: BBox, _: BBox) -> bool:
        return box1.item_obj.start + buffer < box1.frame < box1.item_obj.end - buffer

    return buffer_filter


def boundary_filter(box1: BBox, _: BBox) -> bool:
    """Check whether ``box1`` is on a boundary frame of another bird in the same frame.

    :param box1: Bounding box to test.
    :type box1: BBox
    :param _: Unused second bounding box.
    :type _: BBox
    :return: ``True`` if ``box1`` is on another bird's boundary frame, otherwise ``False``.
    :rtype: bool
    """
    return any(
        box1.frame in (bird.item_obj.start, bird.item_obj.end) for bird in box1.frame_obj.birds if bird is not box1.item_obj
    )


def invert_filter(func: FilterFunc) -> FilterFunc:
    """Return a filter that negates the result of another filter.

    :param func: Filter function to invert.
    :type func: FilterFunc
    :return: A filter function returning the logical negation of ``func``.
    :rtype: FilterFunc
    """

    def inverted_filter(box1: BBox, box2: BBox) -> bool:
        return not func(box1, box2)

    return inverted_filter
