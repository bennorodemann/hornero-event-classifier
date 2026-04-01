"""Core classifier building blocks and shared utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Self,
    Sequence,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from hornero_event_classifier.classifiers.dependencies import Dependency
from hornero_event_classifier.classifiers.metrics import (
    Metric,
    metric_func_registry,
)
from hornero_event_classifier.core import BBox, Item, Subject


class ItemSegment:  # pylint: disable=too-many-instance-attributes
    """A sub-segment of :py:class:`~.core.data.Item` that holds a sub-section :py:class:`~.core.data.BBox`\\es and holds
    :py:class:`~.Metric` data arrays about the :py:class:`~core.data.BBox`\\es. This is the class type that
    :py:class:`Classifier`\\s use. This class should generally be initiated using :py:class:`SegmentCollection`.

    len(:py:class:`ItemSegment`) returns the number of :py:class:`~.core.data.BBox`\\es in the segment

    :param item: parent :py:class:`~core.data.Item`
    :type item: Item
    :param boxes: a sequence of :py:class:`~.core.data.BBox`\\es from the parent
    :type boxes: Sequence[BBox]
    :param metrics: :py:class:`~.classifiers.metrics.Metric`\\s to collect data on
    :type metrics: Iterable[Metric]
    """

    def __init__(self, item: Item, boxes: Sequence[BBox], metrics: Iterable[Metric]) -> None:
        self.item: Item = item
        self.metrics: Iterable[Metric] = metrics
        self.boxes: list[BBox] = list(boxes)
        self.start: int
        self.end: int
        self.seg_data: NDArray[np.float64]
        self.data_summary: NDArray[np.float64]
        self.refresh()
        self.classification: Subject = Subject.NOT_CLASSIFIED

    def __repr__(self) -> str:
        return f"{self.classification}: {self.start} -> {self.end}"

    def __len__(self):
        return len(self.boxes)

    @classmethod
    def combine(cls, source: Sequence[ItemSegment]) -> Self:
        """Merge segments that share the same parent item and classification.

        :param source: Segments to merge (must share parent item and classification)
        :type source: Sequence[ItemSegment]
        :raises ValueError: If segments do not share the same item and classification
        :return: Combined segment
        :rtype: Self
        """
        ref: ItemSegment = source[0]
        if not all(box.classification == ref.classification and box.item is ref.item for box in source):
            raise ValueError("All source Segments must have the same classification and item")
        new = cls(ref.item, [box for segment in source for box in segment.boxes], ref.metrics)
        new.classification = ref.classification
        return new

    @property
    def size(self):
        """Return the inclusive frame span of the segment."""
        return (self.boxes[-1].frame - self.boxes[0].frame) + 1

    @property
    def max_birds(self):
        """Return the maximum number of birds in any frame within the segment."""
        return max(sum(1 for _ in box.frame_obj.birds) for box in self.boxes)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the segment summary to a dictionary."""
        return {
            "item_id": self.item.key,
            "start_frame": self.start,
            "end_frame": self.end,
            **{k.name: v for k, v in zip(self.metrics, self.data_summary)},
        }

    def refresh(self):
        """Recompute cached metric data for the current segment boxes. This is called automatically when data is changed using
        a method."""
        self.boxes.sort()
        self.start = self.boxes[0].frame
        self.end = self.boxes[-1].frame
        data = []
        for metric in self.metrics:
            args = []
            # gather input arguments
            for arg in metric_func_registry.get_args(metric):
                args.append([box.metrics_cache[arg] for box in self.boxes])
            # get metric data (on a per box basis) and append to data
            data.append(metric_func_registry.get(metric)(self.boxes, *args))
        # convert data list to array (rows = box data, columns = metric data)
        self.seg_data: NDArray[np.float64] = np.array(list(zip(*data)), np.float64)
        # new array where metrics that are NaN for all boxes, convert them to 0, otherwise leave as is (this if to avoid warnings)
        no_nan_only = np.where(np.all(np.isnan(self.seg_data), axis=0), 0, self.seg_data)
        # get mean of each metric ignoring NaNs
        self.data_summary: NDArray[np.float64] = np.nanmean(no_nan_only, axis=0)

    def include(self, data: ItemSegment | Iterable[BBox]):
        """Extend this segment with another segment or a sequence of boxes."""
        if isinstance(data, ItemSegment):
            self.boxes.extend(data.boxes)
        else:
            self.boxes.extend(data)
        self.refresh()

    def get_tail_segment(self, count: int) -> ItemSegment:
        """Return a new segment containing only the final ``count`` boxes."""
        out = ItemSegment(self.item, self.boxes[-count:], self.metrics)
        out.classification = self.classification
        return out

    def cut(self, index: int, from_end: bool = True) -> list[BBox]:
        """Cut the segment and return the removed boxes.

        :param index: Cut index (negative values count from the end)
        :type index: int
        :param from_end: If ``True``, cut from the end; otherwise cut from the start
        :type from_end: bool
        :return: Boxes removed from this segment
        :rtype: list[BBox]
        """
        if abs(index) >= len(self):
            raise ValueError(f"Cut index out of range: {index}")
        if from_end:
            out = self.boxes[index:]
            self.boxes = self.boxes[:index]
        else:
            out = self.boxes[:index]
            self.boxes = self.boxes[index:]
        self.refresh()
        return out

    def edit_border(self, other: ItemSegment, offset: int):
        """Shift a shared border between adjacent segments by ``offset`` frames."""
        # check if shares border and normalize order
        if self.end + 1 == other.start:
            prev, next_ = self, other
        elif self.start - 1 == other.end:
            prev, next_ = other, self
        else:
            raise ValueError("No border is shared with provided segment")
        # if no offset do nothing
        if offset == 0:
            return
        # if offset is negative
        if offset < 0:
            try:
                next_.include(prev.cut(offset))
                return
            except ValueError as e:
                raise e from ValueError(f"Offset out of range: {offset}")
        # if offset is positive
        prev.include(next_.cut(offset, from_end=False))


class SegmentCollection:
    """Collection of :py:class:`ItemSegment`\\s built from a set of :py:class:`~.core.data.Item`\\s.

    :param items: Items to segment
    :type items: Iterable[Item]
    :param metrics: Metrics to compute per segment
    :type metrics: Iterable[Metric]
    :param segment_length: Optional fixed length for segmentation, if None (the default) full item length is used
    :type segment_length: Optional[int], optional
    """

    def __init__(self, items: Iterable[Item], metrics: Iterable[Metric], segment_length: Optional[int] = None) -> None:
        self.metrics: tuple[Metric, ...] = tuple(metrics)
        self.target_segment_len: int | None = segment_length
        segments: list[ItemSegment] = []
        self.item_groups: dict[Item, slice] = {}
        items = list(items)
        prev_len: int | None = None
        # get and load metric dependencies
        cache_seq = self._get_cache_sequence(metrics)
        for item in items:
            boxes = tuple(item.boxes.get_all())
            # run dependency functions
            for func in cache_seq:
                func(boxes)
            # load segments and add them to all segments list
            segments.extend(self._load_segments(item, boxes, self.metrics, segment_length=segment_length))
            new_len = len(segments)
            # remember which indexes correspond to which items
            self.item_groups[item] = slice(prev_len, new_len)
            prev_len = new_len
        # lock down segments
        self.segments: tuple[ItemSegment, ...] = tuple(segments)
        # combine all segment data into single array for quick analysis
        self.data: NDArray[np.floating] = np.array([segment.data_summary for segment in segments])

    @staticmethod
    def _load_segments(
        item: Item, boxes: Sequence[BBox], metrics: Iterable[Metric], segment_length: Optional[int]
    ) -> list[ItemSegment]:
        if segment_length is None:
            return [ItemSegment(item, boxes, metrics)]
        return [ItemSegment(item, boxes[s : (s + segment_length - 1)], metrics) for s in range(0, len(boxes), segment_length)]

    @staticmethod
    def _get_cache_sequence(metrics: Iterable[Metric]) -> list[Callable[[Sequence[BBox]], Any]]:
        all_funcs: set[Dependency] = set()
        for metric in metrics:
            all_funcs |= metric_func_registry.get_dependency_list(metric)
        return sorted(all_funcs, key=lambda d: d.order)

    def item_segment_data(self) -> Iterable[NDArray[np.floating]]:
        """Iterate over per-item segment data arrays."""
        return (self.data[slice_] for slice_ in self.item_groups.values())

    def item_segments(self) -> Iterable[tuple[ItemSegment, ...]]:
        """Iterate over per-item segment tuples."""
        return (self.segments[slice_] for slice_ in self.item_groups.values())

    def __getitem__(self, key: Item) -> NDArray[np.floating]:
        """Return the data matrix for a single :py:class:`~core.data.Item`."""
        return self.data[self.item_groups[key]]

    def as_df(self, video_id: str):
        """Return a pandas DataFrame summary of all segments. (For debugging purposes)"""
        data = [seg.as_dict() for seg in self.segments]
        df = pd.DataFrame(data)
        df.insert(0, "video_id", video_id)
        return df


class Classifier(ABC):
    """Abstract base class for classifiers operating on :py:class:`SegmentCollection` data."""

    def __init__(self, metrics: Iterable[Metric]) -> None:
        self.metrics: tuple[Metric, ...] = tuple(metrics)

    def train(self, data: SegmentCollection):
        """Optional training hook for subclasses."""

    def classify(self, data: SegmentCollection) -> dict[Item, Sequence[ItemSegment]]:
        """Classify segments for each item and return :py:meth:`_simplify`\\ed segment groups.

        This method should generally not be overwritten when subclassing.
        """
        if data.metrics != self.metrics:
            raise ValueError(f"The metrics of {type(self).__name__} and {type(data).__name__} do not match.")
        # get classification results
        classifications = self.classify_matrix(data.data)
        # apply classifications to segments
        self._apply_classifications(data.segments, classifications)
        # for group if item segments, if there more than one segment call self.clean_seq
        for item_idx in data.item_groups.values():
            item_segments = data.segments[item_idx]
            if len(item_segments) > 1:
                classifications[item_idx] = self.clean_seq(data.segments[item_idx], classifications[item_idx])
        # apply cleaned classifications
        self._apply_classifications(data.segments, classifications)
        # return dict of condensed segments of consecutive segments with the same classification
        return {item: self._simplify(data.segments[item_idx]) for item, item_idx in data.item_groups.items()}

    @staticmethod
    def _apply_classifications(segments: Iterable[ItemSegment], classifications: Iterable[bool] | NDArray[np.bool]):
        """Apply boolean classifications to a sequence of segments."""
        for segment, classification in zip(segments, classifications):
            segment.classification = Subject(classification)

    def _simplify(self, segments: tuple[ItemSegment, ...]) -> tuple[ItemSegment, ...]:
        """Merge adjacent segments that share the same classification."""
        if len(segments) <= 1:
            return segments
        cur_seq: list[ItemSegment] = [segments[0]]
        segment_seqs: list[list[ItemSegment]] = [cur_seq]
        for segment in segments[1:]:
            if segment.classification == cur_seq[-1].classification:
                cur_seq.append(segment)
            else:
                cur_seq = [segment]
                segment_seqs.append(cur_seq)
        return tuple(ItemSegment.combine(segment_seq) for segment_seq in segment_seqs)

    @abstractmethod
    def classify_matrix(self, matrix: NDArray[np.floating]) -> NDArray[np.bool]:
        """Primary methods that **must** be overwritten for subclasses to initiate. This method takes a matrix of data and
        classifies it.

        :param matrix: Input matrix with the same number of columns as the classifier has :py:class:`Metric`\\s.
        :type matrix: NDArray[np.floating]
        :return: A matrix of boolean of the same length as ``matrix`` rows.
        :rtype: NDArray[np.bool]
        """

    def clean_seq(self, segments: tuple[ItemSegment, ...], raw_classifications: NDArray[np.bool]) -> NDArray[np.bool]:
        """An optional method for post processing of segment classifications.

        This method is called when an item has more than one segment (``segment_length < len(item)``). This method can be used to
        clean noisy classification results.

        By default this method return ``raw_classifications`` unchanged.

        :param segments: All item segments belonging to a specific item.
        :type segments: tuple[ItemSegment, ...]
        :param raw_classifications: The original classifications for provided segments.
        :type raw_classifications: NDArray[np.bool]
        :return: The cleaned boolean array of the same length as ``raw_classifications``
        :rtype: NDArray[np.bool]
        """
        return raw_classifications
