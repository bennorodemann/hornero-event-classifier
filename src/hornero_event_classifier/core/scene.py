import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, Iterable, Optional
import pandas as pd

from hornero_event_classifier.classifiers import Classifier, SegmentCollection
from hornero_event_classifier.core.data import BBox, Frame, Item
from hornero_event_classifier.core.enums import ItemType, Subject
from hornero_event_classifier.core.filters import FilterFunc, boundary_filter
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
    filename: str
    source: Path
    items: ItemTypedCollection[Item] = field(repr=False)
    frames: FrameIndexer[Frame] = field(repr=False)

    @classmethod
    def from_csv(cls, filepath: str | Path) -> Self:
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"file not found: {filepath}")
        with open("databases/general/video_metadata.csv", "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for vid in reader:
                if vid["video"] == filepath.stem.rsplit("_", 1)[0]:
                    frame_shape = (int(vid["height"]), int(vid["width"]))
                    break
            else:
                raise ValueError(f"No video found in metadata with name: {filepath.stem.rsplit("_", 1)[0]}")
        items: DefaultSpawnDict[str, Item] = DefaultSpawnDict(_item_read_spawner)
        frames: DefaultSpawnDict[int, Frame] = DefaultSpawnDict(Frame, defaults={"frame_shape": frame_shape})
        with open(filepath, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for data in reader:
                typed_row: YOLOData = type_yolo_data(data)
                BBox.from_yolo(typed_row, items[typed_row["ID"]], frames[typed_row["Frame"]])
        for item in items.values():
            item.release_cache()
        return cls(
            filename=filepath.name,
            source=filepath.parent,
            items=ItemTypedCollection(items.values()),
            frames=FrameIndexer(frames),
        )

    @property
    def video_id(self) -> str:
        return self.filename.rsplit("_", 1)[0]

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

    def fill_gaps(self, filter_func: Optional[FilterFunc | Iterable[FilterFunc]] = None, *item_types: ItemType) -> Self:
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
                        self.frames[frame] = Frame(frame)
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
    def merge_birds(self, overlap: float, correlation: float, exists_only: bool = False) -> Self:
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
                # print(f"destroying: {item}")
                item.ignore = True
                # item.destroy()
                # self.items.remove(item)
        return self

    # def split_at_boundary(self, buffer: int = 0) -> Self:
    #     cuts = {frame for item in self.items.get(ItemType.BIRD) for frame in (item.start, item.end + 1)}
    #     for frame in cuts:
    #         frame_obj = self.frames.get(frame, None)
    #         if frame_obj is not None:
    #             for bird in frame_obj.birds:
    #                 if bird.item_obj.start + buffer < frame < bird.item_obj.end - buffer:
    #                     new = bird.item_obj.cut_at(frame)
    #                     self.items.add(new)
    #     return self

    # def split_at_frame_touch(self, buffer: int = 0, max_touch_time: int = 30, merge: int = 30) -> Self:
    #     ref_frame = self.frames[self.frames.start]
    #     frame_w = ref_frame.width - 5
    #     frame_h = ref_frame.height - 5
    #     for bird in [b for b in self.items.get(ItemType.BIRD)]:
    #         if (bird.end - bird.start + 1) <= buffer * 2:
    #             continue
    #         cuts: list[int] = []
    #         boxes = [box for box in bird.boxes.get_all() if bird.start + buffer < box.frame < bird.end - buffer]
    #         length_counter: int = 0
    #         merge_counter: int = merge
    #         for box in boxes:
    #             touching = ((box.xmin < 5 or box.xmax > frame_w) and box.width < frame_w / 2) or (
    #                 (box.ymin < 5 or box.ymax > frame_h) and box.height < frame_h / 2
    #             )
    #             if touching:
    #                 length_counter += 1
    #                 merge_counter = merge
    #             elif length_counter > 0:
    #                 merge_counter -= 1
    #                 if merge_counter == 0:
    #                     if length_counter <= max_touch_time:
    #                         cuts.append(box.frame)
    #                     length_counter = 0
    #                     merge_counter = merge
    #         for frame in sorted(cuts, reverse=True):
    #             new = bird.cut_at(frame)
    #             self.items.add(new)
    #     return self

    def classify(self, classifier: Classifier, segment_length: Optional[int] = None) -> Self:
        data = SegmentCollection(self.items.get(ItemType.BIRD), classifier.metrics, segment_length=segment_length)
        classifier.train(data)
        results = classifier.classify(data)
        for item, segments in results.items():
            item.subject = segments[0].classification
            for segment in segments[:0:-1]:
                new_item = item.cut_at(segment.start)
                new_item.subject = segment.classification
                self.items.add(new_item)
        return self

    def define_events(self, buffer: int = 0, micro_events: int = 0) -> Self:
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
        # for event in new_events.copy():
        #     if event.track_len > micro_events:
        #         continue
        #     for other_event in [e for e in new_events if e is not event and not e.ignore]:
        #         if event is not other_event and event.start > other_event.start - buffer and event.end < other_event.end + buffer:
        #             event.subject = other_event.subject
        #             new_events.append(Item.combine([event, other_event]))
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
        results.groupby()
        # with open(f"databases/pYOLOv3/{self.video_id}_events.csv", "w", encoding="utf-8") as file:
        #     writer = csv.DictWriter(file, ("video_id", "subject", "start", "end", "mud"))
        #     writer.writeheader()
        #     for event in self.items.get(ItemType.EVENT):
        #         if not allow_no_subject and event.subject != Subject.NOT_CLASSIFIED:
        #             raise ValueError("Tried to save an event without a subject")
        #         writer.writerow(
        #             {
        #                 "video_id": self.video_id,
        #                 "subject": event.subject.value,
        #                 "start": event.start,
        #                 "end": event.end,
        #                 "mud": False,
        #             }
        #         )
        return self
