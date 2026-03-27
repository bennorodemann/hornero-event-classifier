import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, Iterable, Optional
import pandas as pd
import warnings

from hornero_event_classifier.classifiers import Classifier, SegmentCollection
from hornero_event_classifier.core.data import BBox, Frame, Item
from hornero_event_classifier.core.enums import ItemType, Subject
from hornero_event_classifier.core.filters import FilterFunc, boundary_filter
from hornero_event_classifier.tools import get_video_metadata, get_video_id
from hornero_event_classifier.core.utils import (
    DefaultSpawnDict,
    FrameIndexer,
    YOLOData,
    compare_seq,
    type_yolo_data,
    ItemTypedCollection,
    ResultDict,
)


def _item_read_spawner(key: str) -> Item:
    new = Item.from_str(key)
    new.start_caching()
    return new


@dataclass
class Scene:
    video_id: str
    frame_shape: tuple[int, int]
    items: ItemTypedCollection[Item] = field(default_factory=ItemTypedCollection[Item], repr=False)
    frames: FrameIndexer[Frame] = field(default_factory=FrameIndexer, repr=False)
    segments: SegmentCollection | None = field(default=None, repr=False, init=False)

    @classmethod
    def from_csv(cls, filepath: str | Path) -> Self:
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"file not found: {filepath}")
        video_id = get_video_id(filepath)
        metadata = get_video_metadata(video_id)
        frame_shape = (int(metadata["height"]), int(metadata["width"]))
        inst = cls(video_id, frame_shape)
        items: DefaultSpawnDict[str, Item] = DefaultSpawnDict(_item_read_spawner)
        frames: DefaultSpawnDict[int, Frame] = DefaultSpawnDict(Frame, defaults={"_scene": inst})
        with open(filepath, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for data in reader:
                typed_row: YOLOData = type_yolo_data(data)
                BBox.from_yolo(typed_row, items[typed_row["ID"]], frames[typed_row["Frame"]])
        for item in items.values():
            item.release_cache()
        inst.items.extend(items.values())
        inst.frames.include_many(frames.values())
        return inst

    def remove_low_conf(self, threshold: float, *item_types: ItemType) -> Self:
        for item in self.items.get(*item_types):
            conf = [box.conf for box in item.boxes.get_all()]
            if sum(conf) / len(conf) < threshold:
                item.ignore = True

        return self

    def _combine_filters(self, funcs: tuple[FilterFunc, ...]) -> FilterFunc:
        def combo_filter(box1: BBox, box2: BBox) -> bool:
            return all(func(box1, box2) for func in funcs)

        return combo_filter

    def split_items(self, filter_func: FilterFunc | Iterable[FilterFunc], *item_types: ItemType) -> Self:
        if isinstance(filter_func, Iterable):
            original = filter_func
            filter_func = self._combine_filters(tuple(filter_func))
        else:
            original = (filter_func,)
        for item in list(self.items.get(*item_types)):
            cut_frames: list[int] = []
            for prev_box, next_box in compare_seq(item.boxes):
                if filter_func(prev_box, next_box):
                    if boundary_filter in original:
                        pass
                    cut_frames.append(next_box.frame)
            for cut_frame in cut_frames:
                item = item.cut_at(cut_frame)
                assert item not in self.items
                self.items.add(item)
        return self

    def fill_gaps(self, filter_func: Optional[FilterFunc | Iterable[FilterFunc]], *item_types: ItemType) -> Self:
        filter_func = filter_func or ()
        if isinstance(filter_func, Iterable):
            filter_func = self._combine_filters(tuple(filter_func))
        for item in self.items.get(*item_types):
            for prev_box, next_box in item.get_gaps(1):
                if not filter_func(prev_box, next_box):
                    continue
                span: int = next_box.frame - prev_box.frame
                assert prev_box.frame < next_box.frame
                assert span > 1
                for step, frame in enumerate(range(prev_box.frame + 1, next_box.frame), 1):
                    assert step < span
                    pos = step / span
                    xmin = prev_box.xmin - ((prev_box.xmin - next_box.xmin) * pos)
                    xmax = prev_box.xmax - ((prev_box.xmax - next_box.xmax) * pos)
                    ymin = prev_box.ymin - ((prev_box.ymin - next_box.ymin) * pos)
                    ymax = prev_box.ymax - ((prev_box.ymax - next_box.ymax) * pos)
                    conf = prev_box.conf - ((prev_box.conf - next_box.conf) * pos)
                    if not self.frames.has(frame):
                        self.frames[frame] = Frame(frame, self)
                    frame_obj = self.frames[frame]
                    BBox(
                        frame_obj=frame_obj,
                        item_obj=item,
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax,
                        conf=conf,
                        real=False,
                    )
        return self

    # TODO: implement filters (technically deprecated)
    # TODO: reimplement using filters (need to implement item filters)
    def merge_birds(self, overlap: float, correlation: float, exists_only: bool = False) -> Self:
        warnings.warn("Scene.merge_birds is no longer being developed and may lead to unexpected behavior")
        birds: list[Item] = sorted(self.items.get(ItemType.BIRD), key=lambda i: i.start)
        overlaps: dict[Item, set[Item]] = {}
        for bird in birds:
            overlaps[bird] = {b for b in birds if b is not bird and b.frame_overlap(bird) and b.subject == bird.subject}
        for parent_bird, child_birds in overlaps.items():
            for child_bird in child_birds.copy():
                overlap_start = max(parent_bird.start, child_bird.start)
                overlap_end = min(parent_bird.end, child_bird.end)
                correlation_counter = 0
                overlap_counter = 0
                for bbox in parent_bird.boxes[overlap_start:overlap_end]:
                    if child_bird.boxes.has(bbox.frame):
                        overlap_counter += 1
                        if bbox.overlap_with(child_bird.boxes[bbox.frame]) >= overlap:
                            correlation_counter += 1
                frame_count: int
                if exists_only:
                    frame_count = overlap_counter
                else:
                    frame_count = overlap_end - overlap_start + 1
                if (correlation_counter / frame_count) < correlation:
                    child_birds.remove(child_bird)
                    overlaps[child_bird].remove(parent_bird)
        new_items: list[tuple[Item, ...]] = []
        while overlaps:
            parent, children = overlaps.popitem()
            new_item_children = {parent, *children}
            if not children:
                continue
            queue = children
            while queue:
                more_children = overlaps.pop(queue.pop(), None)
                if more_children is None:
                    continue
                new_item_children |= more_children
                queue |= new_item_children - more_children
            new_items.append(tuple(new_item_children))
        for new_item_children in new_items:
            new_item = Item.combine(new_item_children)
            self.items.add(new_item)
            for child in new_item_children:
                child.ignore = True
        return self

    def remove_minor_items(self, size: int, *item_types: ItemType) -> Self:
        for item in list(self.items.get(*item_types)):
            if len(item.boxes) <= size:
                item.ignore = True
        return self

    def classify(self, classifier: Classifier, segment_length: Optional[int] = None) -> Self:
        self.segments = SegmentCollection(self.items.get(ItemType.BIRD), classifier.metrics, segment_length=segment_length)
        classifier.train(self.segments)
        results = classifier.classify(self.segments)
        for item, segments in results.items():
            item.subject = segments[0].classification
            for segment in segments[:0:-1]:
                new_item = item.cut_at(segment.start)
                new_item.subject = segment.classification
                self.items.add(new_item)
        return self

    def define_events(self, buffer: int = 0) -> Self:
        cache: list[list[Item]] = []
        active: dict[Subject, list[Item]] = {}
        for item in sorted(self.items.get(ItemType.BIRD), key=lambda b: b.start):
            assert item.subject is not None
            item_group: list[Item] | None = active.get(item.subject, None)
            if item_group is None:
                active[item.subject] = [item]
                continue
            if len(item_group) == 0 or any((i.end + buffer) >= item.start for i in item_group):
                item_group.append(item)
                continue
            cache.append(item_group)
            active[item.subject] = [item]
        for group in active.values():
            cache.append(group)
        new_events: list[Item] = []
        for id_, event_data in enumerate(cache, 1):
            event = Item.spawn_event(id_=id_, source=event_data)
            new_events.append(event)
        for event in new_events:
            self.items.add(event)
        return self

    def _get_result(self, item: Item) -> ResultDict:
        return {
            "video_id": self.video_id,
            "subject": item.subject.value,
            "start_frame": item.start,
            "end_frame": item.end,
            "mud": False,
        }

    def get_results(self, validate: bool = True) -> pd.DataFrame:
        events = list(self.items.get(ItemType.EVENT))
        if validate and any(event.subject == Subject.NOT_CLASSIFIED for event in events):
            raise ValueError("Results contain an event without as subject")
        return pd.DataFrame([self._get_result(event) for event in events])

    def write_to_csv(self, filename: str = "", allow_no_subject: bool = False) -> Self:
        results = self.get_results(not allow_no_subject)
        results.to_csv(filename, index=False)
        return self
