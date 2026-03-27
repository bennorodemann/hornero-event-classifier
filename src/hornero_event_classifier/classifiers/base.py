from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Self,
    Sequence,
)
import warnings

import numpy as np
import pandas as pd
from hornero_event_classifier.classifiers.metrics import (
    Metric,
    metric_func_registry,
)
from hornero_event_classifier.classifiers.pre_calc import Dependency
from hornero_event_classifier.core import BBox, Item, Subject
from numpy.typing import NDArray


class ItemSegment:
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item.key,
            "start_frame": self.start,
            "end_frame": self.end,
            **{k.name: v for k, v in zip(self.metrics, self.data_summary)},
        }  # type: ignore

    def refresh(self):
        self.boxes.sort()  # key=lambda b: b.frame_obj.frame)
        self.start = self.boxes[0].frame
        self.end = self.boxes[-1].frame
        data = []
        for metric in self.metrics:
            args = []
            for arg in metric_func_registry.get_args(metric):
                args.append([box.metrics_cache[arg] for box in self.boxes])
            data.append(metric_func_registry.get(metric)(self.boxes, *args))
        self.seg_data: NDArray[np.float64] = np.array(list(zip(*data)), np.float64)  # - 0.5
        # self.seg_data = np.nan_to_num(self.seg_data)
        # self.data_summary = np.mean(self.seg_data, axis=0)
        self.seg_data = np.where(np.all(np.isnan(self.seg_data), axis=0), 0, self.seg_data)
        self.data_summary: NDArray[np.float64] = np.nanmean(self.seg_data, axis=0)

        # self.data_summary = np.where(np.all(np.isnan(self.seg_data), axis=0), 0, np.nanmean(self.seg_data, axis=0))
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=RuntimeWarning)
        # self.data_summary: NDArray[np.float64] = np.nanmean(self.seg_data, axis=0)
        # self.data_summary = np.nan_to_num(self.data_summary)

    def __len__(self):
        return len(self.boxes)

    @classmethod
    def combine(cls, source: Sequence[ItemSegment]) -> Self:
        ref: ItemSegment = source[0]
        if not all(box.classification == ref.classification and box.item is ref.item for box in source):
            raise ValueError("All source Segments must have the same classification and item")
        new = cls(ref.item, [box for segment in source for box in segment.boxes], ref.metrics)
        new.classification = ref.classification
        return new

    @property
    def size(self):
        return (self.boxes[-1].frame - self.boxes[0].frame) + 1

    @property
    def max_birds(self):
        return max(sum(1 for _ in box.frame_obj.birds) for box in self.boxes)

    def include(self, data: ItemSegment | Iterable[BBox]):
        if isinstance(data, ItemSegment):
            self.boxes.extend(data.boxes)
        else:
            self.boxes.extend(data)
        self.refresh()

    def get_tail_segment(self, count: int) -> ItemSegment:
        out = ItemSegment(self.item, self.boxes[-count:], self.metrics)
        out.classification = self.classification
        return out

    def cut(self, index: int, from_end: bool = True) -> list[BBox]:
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
            except ValueError as e:
                raise e from ValueError(f"Offset out of range: {offset}")
        # if offset is positive
        prev.include(next_.cut(offset, from_end=False))


class SegmentCollection:
    def __init__(self, items: Iterable[Item], metrics: Iterable[Metric], segment_length: Optional[int] = None) -> None:
        self.metrics: tuple[Metric, ...] = tuple(metrics)
        self.target_segment_len: int | None = segment_length
        segments: list[ItemSegment] = []
        self._item_groups: dict[Item, slice] = {}
        items = list(items)
        prev_len: int | None = None
        cache_seq = self._get_cache_sequence(metrics)
        for item in items:
            boxes = tuple(item.boxes.get_all())
            for func in cache_seq:
                func(boxes)
            segments.extend(self._load_segments(item, boxes, self.metrics, segment_length=segment_length))
            new_len = len(segments)
            self._item_groups[item] = slice(prev_len, new_len)
            prev_len = new_len
        self.segments: tuple[ItemSegment, ...] = tuple(segments)
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
        return (self.data[self._item_groups[item]] for item in self._item_groups)

    def item_segments(self) -> Iterable[tuple[ItemSegment, ...]]:
        return (self.segments[self._item_groups[item]] for item in self._item_groups)

    def __getitem__(self, key: Item):
        return self.data[self._item_groups[key]]

    def as_df(self, video_id: str):  # TODO: remove?
        data = [seg.as_dict() for seg in self.segments]
        df = pd.DataFrame(data)
        df.insert(0, "video_id", video_id)
        return df


class Classifier(ABC):
    def __init__(self, metrics: Iterable[Metric]) -> None:
        self.metrics: tuple[Metric, ...] = tuple(metrics)

    def train(self, data: SegmentCollection):
        pass

    def classify(self, data: SegmentCollection) -> dict[Item, Sequence[ItemSegment]]:
        classifications = self.classify_matrix(data.data)
        self._apply_classifications(data.segments, classifications)
        for item_idx in data._item_groups.values():
            if item_idx.stop is not None and item_idx.start is not None and item_idx.stop - item_idx.start <= 1:
                continue
            classifications[item_idx] = self.clean_seq(data.segments[item_idx], classifications[item_idx])
        self._apply_classifications(data.segments, classifications)
        return {item: self._simplify(data.segments[item_idx]) for item, item_idx in data._item_groups.items()}

    @staticmethod
    def _apply_classifications(segments: Iterable[ItemSegment], classifications: Iterable[bool] | NDArray[np.bool]):
        for segment, classification in zip(segments, classifications):
            segment.classification = Subject(classification)

    def _simplify(self, segments: tuple[ItemSegment, ...]) -> tuple[ItemSegment, ...]:
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
    def classify_matrix(self, matrix: NDArray[np.floating]) -> NDArray[np.bool]: ...

    @abstractmethod
    def clean_seq(self, segment: tuple[ItemSegment, ...], raw_classifications: NDArray[np.bool]) -> NDArray[np.bool]: ...

    def smooth(self, classifications: NDArray[np.bool]) -> NDArray[np.bool]:
        return classifications

    def _calculate_border_offset(self, segment1: ItemSegment, segment2: ItemSegment) -> int:
        return 0

    def save_segment_csv(self, video_id: str, data: SegmentCollection):
        with open(f"databases/blocks/{video_id}.csv", "w", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, ("video_id", "item_id", "segment_id", "frame", "type", "xmin", "xmax", "ymin", "ymax", "conf")
            )
            writer.writeheader()
            for segment_id, segment in enumerate(data.segments, 1):
                for box in segment.boxes:
                    writer.writerow(
                        {
                            "video_id": video_id,
                            "item_id": segment.item.key,
                            "segment_id": segment_id,
                            "frame": box.frame,
                            "type": segment.item.type.value,
                            "xmin": box.xmin,
                            "xmax": box.xmax,
                            "ymin": box.ymin,
                            "ymax": box.ymax,
                            "conf": box.conf,
                        }
                    )
