"""Scene construction and processing for a single video.

The :py:class:`Scene` class wraps the full pipeline for loading YOLO detections, organizing them into items and frames, running
classifiers, and exporting results. It is the main orchestration object used by higher-level workflows.

Example
-------
.. code-block:: python3

    from hornero_event_classifier.core.video_metadata import VideoMetadata
    from hornero_event_classifier.core.scene import Scene

    metadata = VideoMetadata(...)
    scene = Scene.from_metadata(metadata)
    scene.remove_low_conf(0.2)
    scene.fill_gaps(None)
    scene.classify(...)
    results = scene.get_results()
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Self

import pandas as pd

from hornero_event_classifier.classifiers import Classifier, SegmentCollection
from hornero_event_classifier.core.collections import (
    DefaultSpawnDict,
    FrameIndexer,
    ItemTypedCollection,
)
from hornero_event_classifier.core.data import BBox, Frame, Item
from hornero_event_classifier.core.enums import ItemType, Subject
from hornero_event_classifier.core.filters import FilterFunc
from hornero_event_classifier.core.types import ResultDict, YOLOData, type_yolo_data

if TYPE_CHECKING:
    from hornero_event_classifier.core.video_metadata import VideoMetadata


def _item_read_spawner(key: str) -> Item:
    new = Item.from_str(key)
    new.start_caching()
    return new


@dataclass
class Scene:
    """Container for all data and processing steps for a single video.

    A ``Scene`` holds:
        - the :py:class:`.VideoMetadata` for the video,
        - a collection of :py:class:`.Item` objects,
        - a frame index of :py:class:`.Frame` objects,
        - and classification :py:class:`.SegmentCollection` data, which starts as ``None`` and is set once \
            :py:meth:`Scene.classify` is called.

    Most methods return the current ``Scene`` object, allowing for method chaining:

    .. code-block:: python

        results = (
            Scene.from_metadata(...)
            .remove_low_conf(...)
            .fill_gaps(...)
            .classify(...)
            .get_results()
        )

    Scenes are typically built via :py:meth:`Scene.from_metadata` which reads YOLO CSV detections and constructs items and
    frames.
    """

    video_data: VideoMetadata
    items: ItemTypedCollection[Item] = field(default_factory=ItemTypedCollection[Item], repr=False)
    frames: FrameIndexer[Frame] = field(default_factory=FrameIndexer, repr=False)
    segments: SegmentCollection | None = field(default=None, repr=False, init=False)

    @classmethod
    def from_metadata(cls, metadata: VideoMetadata) -> Self:
        """Primary ``Scene`` constructor.

        :param metadata: Source video's :py:class:`.VideoMetadata` object.
        :type metadata: VideoMetadata
        :raises FileNotFoundError: Raise an error if :py:attr:`.VideoMetadata.yolo_path` could not be found.
        :return: Loaded instance of video YOLO data.
        :rtype: Self
        """
        if not metadata.yolo_path.is_file():
            raise FileNotFoundError(f"file not found: {metadata.yolo_path}")
        # create instance
        inst = cls(metadata)
        # dicts auto-spawn new instances of objects if missing
        items: DefaultSpawnDict[str, Item] = DefaultSpawnDict(_item_read_spawner)
        frames: DefaultSpawnDict[int, Frame] = DefaultSpawnDict(Frame, defaults={"video_metadata": metadata})
        # read csv file
        with open(metadata.yolo_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for data in reader:
                # convert string values to correct types
                typed_row: YOLOData = type_yolo_data(data)
                # spawn new box (auto-attaches to corresponding item and frame objects)
                BBox.from_yolo(typed_row, items[typed_row["ID"]], frames[typed_row["Frame"]])
        # release bbox cache of all items
        for item in items.values():
            item.release_cache()
        # add created items and frames to created instance
        inst.items.extend(items.values())
        inst.frames.include_many(frames.values())
        return inst

    def remove_low_conf(self, threshold: float, *item_types: ItemType) -> Self:
        """Mark low-confidence items as ignored.

        Sets ``Item.ignore = True`` for :py:class:`.Item`\\s of types in ``item_types`` whose average
        :py:class:`BBox` confidence is below ``threshold``. If no ``item_types`` are provided, all types are included.

        :param threshold: Minimum threshold of :py:class:`.Item` to keep in scene.
        :type threshold: float
        :param item_types: One or more :py:class:`.ItemType`\\s to filter. If none are passed, all types are included.
        :type item_types: ItemType
        :return: This :py:class:`Scene` instance.
        :rtype: Self
        """
        # per item of item types in item_types
        for item in self.items.get(*item_types):
            # collect all confidence values of bboxes in item
            conf = [box.conf for box in item.boxes.get_all()]
            # check if average confidence is below threshold
            if sum(conf) / len(conf) < threshold:
                item.ignore = True

        return self

    def _combine_filters(self, funcs: tuple[FilterFunc, ...]) -> FilterFunc:
        # return a function that ensures all funcs return True
        def combo_filter(box1: BBox, box2: BBox) -> bool:
            return all(func(box1, box2) for func in funcs)

        return combo_filter

    def split_items(self, filter_func: FilterFunc | Iterable[FilterFunc], *item_types: ItemType) -> Self:
        """Split items when consecutive :py:class:`.BBox`\\s match a filter.

        This is only applied to :py:class:`.Item`\\s of :py:class:`.ItemType`\\s in ``item_types`` (if none are passed then all
        :py:class:`.ItemType`\\s are included).

        This method applies ``filter_func`` to each pair of consecutive :py:class:`.BBox` pairs in an :py:class:`.Item`. If
        ``filter_func`` returns ``True`` the :py:class:`.Item` is cut at that point, producing a new child :py:class:`.Item`.

        :param filter_func: Filter function(s) to apply. If multiple are passed then they are combined in a logical AND.
        :type filter_func: FilterFunc | Iterable[FilterFunc]
        :param item_types: One or more :py:class:`.ItemType`\\s to filter. If none are passed, all types are included.
        :type item_types: ItemType
        :return: This :py:class:`Scene` instance.
        :rtype: Self
        """
        # if there are multiple filter functions, combine them into a single function (AND-wise)
        if isinstance(filter_func, Iterable):
            filter_func = self._combine_filters(tuple(filter_func))
        # for every item of type in item_types
        for item in list(self.items.get(*item_types)):
            cut_frames: list[int] = []
            data: list[BBox] = list(item.boxes.get_all())
            # compare sequential bboxes
            for prev_box, next_box in zip(data, data[1:]):
                # if filter func returns True, log that item should be cut at next_box.frame
                if filter_func(prev_box, next_box):
                    cut_frames.append(next_box.frame)
            # for all logged cut frames: cut at frame, ensure item ID is new (debugging), add item to scene items
            for cut_frame in cut_frames:
                item = item.cut_at(cut_frame)
                assert item not in self.items
                self.items.add(item)
        return self

    def fill_gaps(self, filter_func: Optional[FilterFunc | Iterable[FilterFunc]], *item_types: ItemType) -> Self:
        """Fill missing frames with interpolated :py:class:`BBox`\\s.

        This is only applied to :py:class:`.Item`\\s of :py:class:`.ItemType`\\s in ``item_types`` (if none are passed then all
        :py:class:`.ItemType`\\s are included).

        This method applies ``filter_func`` to each pair of consecutive :py:class:`.BBox` pairs separated by 1 or more missing
        frames in an :py:class:`.Item`. If ``filter_func`` returns ``True``, missing frames are filled with linearly
        interpolated :py:class:`BBox`\\s.

        :param filter_func: Filter function(s) to apply. If multiple are passed then they are combined in a logical AND. If
            ``None`` all gaps are filled.
        :type filter_func: Optional[FilterFunc  |  Iterable[FilterFunc]]
        :param item_types: One or more :py:class:`.ItemType`\\s to filter. If none are passed, all types are included.
        :type item_types: ItemType
        :return: This :py:class:`Scene` instance.
        :rtype: Self
        """
        # if none set to empty tuple (will be passed to self._combine_filters and always return True)
        filter_func = filter_func or ()
        # if there are multiple filter functions, combine them into a single function (AND-wise)
        if isinstance(filter_func, Iterable):
            filter_func = self._combine_filters(tuple(filter_func))
        # for every item of type in item_types
        frame_cache = self.frames.get_cache()
        for item in self.items.get(*item_types):
            # for bbox pairs that have any missing frames between them within an item
            for prev_box, next_box in item.get_gaps(1):
                assert prev_box.frame < next_box.frame  # debug step: make sure order is correct
                # if filter is False skip to next bbox pair
                if not filter_func(prev_box, next_box):
                    continue
                # frame difference
                span: int = next_box.frame - prev_box.frame
                assert span > 1  # debug step: make sure there are missing frames
                # calculate differences between bboxes
                xmin_shift = next_box.xmin - prev_box.xmin
                xmax_shift = next_box.xmax - prev_box.xmax
                ymin_shift = next_box.ymin - prev_box.ymin
                ymax_shift = next_box.ymax - prev_box.ymax
                conf_shift = next_box.conf - prev_box.conf

                # for every frame create new box linearly interpolated
                for step, frame in enumerate(range(prev_box.frame + 1, next_box.frame), 1):
                    assert step < span  # debug step: do not recreate ending bbox
                    pos = step / span
                    xmin = prev_box.xmin + (xmin_shift * pos)
                    xmax = prev_box.xmax + (xmax_shift * pos)
                    ymin = prev_box.ymin + (ymin_shift * pos)
                    ymax = prev_box.ymax + (ymax_shift * pos)
                    conf = prev_box.conf + (conf_shift * pos)
                    # if frame has not been made yet create new Frame object and cache it
                    if not self.frames.has(frame):
                        frame_obj = Frame(frame, self.video_data)
                        frame_cache.include(frame_obj)
                    else:
                        frame_obj = self.frames[frame]
                    # create new BBox item
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
            # release any cached new frame so the next Item can use them
            frame_cache.release()
        return self

    def _merge_birds(self, overlap: float, correlation: float, exists_only: bool = False) -> Self:
        # TODO: This method is currently outdated/deprecated and should be reworked to follow the filter system.
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
                        if max(bbox.overlap_with(child_bird.boxes[bbox.frame])) >= overlap:
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

    def remove_minor_items(self, min_size: int, *item_types: ItemType) -> Self:
        """Mark short items as ignored.

        Sets ``Item.ignore = True`` for :py:class:`.Item`\\s of types in ``item_types`` if ``len(Item) < min_size``.

        :param min_size: Minimum length of :py:class:`.Item` to retain
        :type min_size: int
        :param item_types: One or more :py:class:`.ItemType`\\s to filter. If none are passed, all types are included.
        :type item_types: ItemType
        :return: This :py:class:`Scene` instance.
        :rtype: Self
        """
        for item in list(self.items.get(*item_types)):
            if len(item.boxes) < min_size:
                item.ignore = True
        return self

    def classify(self, classifier: Classifier, segment_length: Optional[int] = None) -> Self:
        """Apply a classifier to :py:attr:`~.ItemType.BIRD` :py:class:`.Item`\\s.

        :param classifier: A :py:class:`.Classifier` instance to apply to :py:class:`Item`\\s
        :type classifier: Classifier
        :param segment_length: :py:class:`.SegmentCollection` ``segment_length`` argument, defaults to None
        :type segment_length: Optional[int], optional
        :return: This :py:class:`Scene` instance.
        :rtype: Self
        """
        # create segments from bird items
        self.segments = SegmentCollection(self.items.get(ItemType.BIRD), classifier.metrics, segment_length=segment_length)
        # train classifier if needed
        classifier.train(self.segments)
        # get classifications
        results = classifier.classify(self.segments)
        # for each item and segments pair
        for item, segments in results.items():
            # set items subject to the first segments classification
            item.subject = segments[0].classification
            # go through the rest of the segments in reverse order and cut at start of segment
            for segment in segments[:0:-1]:
                new_item = item.cut_at(segment.start)
                new_item.subject = segment.classification
                self.items.add(new_item)
        return self

    def define_events(self, buffer: int = 0) -> Self:
        """Create events from classified :py:attr:`.ItemType.BIRD` :py:class:`.Item`\\s.

        :param buffer: Number of frames within which to merge :py:class:`.Item`\\s with the same classification, defaults to 0.
        :type buffer: int, optional
        :return: This :py:class:`Scene` instance.
        :rtype: Self
        """
        cache: list[list[Item]] = []
        active: dict[Subject, list[Item]] = {}
        # For each bird item
        for item in sorted(self.items.get(ItemType.BIRD), key=lambda b: b.start):
            assert item.subject is not Subject.NOT_CLASSIFIED  # debug check: make sure item was classified
            # get current grouping of items with the same subject
            item_group: list[Item] | None = active.get(item.subject, None)
            # if there is no active item_group create a new one and move on to next iteration
            if item_group is None:
                active[item.subject] = [item]
                continue
            # if item_group is empty or item starts before end (+ buffer) of any of the items in the group, then add item to group
            if len(item_group) == 0 or any((i.end + buffer) >= item.start for i in item_group):
                item_group.append(item)
                continue
            # current item does not fit in any of the current groups
            # close current group if items with same subject and add cache the group
            cache.append(item_group)
            # create a new group with current item in it
            active[item.subject] = [item]
        # close remaining groups and add them to the cache
        for group in active.values():
            cache.append(group)
        new_events: list[Item] = []
        # create a new event from each of the groups
        for id_, event_data in enumerate(cache, 1):
            event = Item.spawn_event(id_=id_, source=event_data)
            new_events.append(event)
        # add event Item instances to scene
        for event in new_events:
            self.items.add(event)
        return self

    def _get_result(self, item: Item) -> ResultDict:
        # convert Item to result csv dict entry
        return {
            "video_id": self.video_data.name,
            "subject": item.subject.value,
            "start_frame": item.start,
            "end_frame": item.end,
            "mud": False,
        }

    def get_results(self) -> pd.DataFrame:
        """Generate a ``pandas.DataFrame`` from created :py:attr:`.ItemType.Event` :py:class:`.Item`\\s.

        The output mimics the column layout from BORIS.

        Columns:
            - video_id: video name string
            - subject: "ring" or "no_ring"
            - start_frame: first frame in event
            - end_frame: last frame in event
            - mud: always ``False``

        :return: Dataframe of found events.
        :rtype: pd.DataFrame
        """
        events = list(self.items.get(ItemType.EVENT))
        return pd.DataFrame([self._get_result(event) for event in events])

    def write_to_csv(self, file_path: str | Path) -> Self:
        """Write dataframe from :py:meth:`Scene.get_results` directly to a CSV file.

        :param file_path: Path to csv file.
        :type file_path: str | Path
        :return: This :py:class:`Scene` instance.
        :rtype: Self
        """
        results = self.get_results()
        results.to_csv(file_path, index=False)
        return self
