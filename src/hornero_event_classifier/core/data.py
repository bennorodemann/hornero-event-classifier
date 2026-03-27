from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Iterable,
    Self,
    Sequence,
)

from hornero_event_classifier.core.utils import FrameCache, FrameIndexer, IDDistributor, YOLOData, ItemTypedCollection, ResultDict

from hornero_event_classifier.core.enums import ItemType, Subject

if TYPE_CHECKING:
    from hornero_event_classifier.core.scene import Scene


@dataclass
class BBox:
    frame_obj: Frame
    item_obj: Item
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    conf: float
    real: bool = True
    metrics_cache: dict = field(default_factory=dict)
    frame: int = field(init=False)

    def __post_init__(self):
        self.frame = self.frame_obj.frame
        self.frame_obj.add_bbox(self)
        self.item_obj.add_bbox(self)

    def __lt__(self, other: BBox):
        return self.frame < other.frame

    def __le__(self, other: BBox):
        return self.frame <= other.frame

    def __gt__(self, other: BBox):
        return self.frame > other.frame

    def __ge__(self, other: BBox):
        return self.frame >= other.frame

    def __eq__(self, other: BBox):
        return self.frame == other.frame

    def __repr__(self) -> str:
        return f"BBox({self.item_obj.key}, {self.frame})"

    def __hash__(self) -> int:
        return hash((type(self), self.frame, self.item_obj))

    @classmethod
    def from_yolo(cls, data: YOLOData, item: Item, frame: Frame) -> Self:
        return cls(frame, item, data["Xmin"], data["Xmax"], data["Ymin"], data["Ymax"], data["Conf"])

    @classmethod
    def surround(cls, item: Item, bboxes: list[BBox] | tuple[BBox, ...]) -> Self:
        ref = bboxes[0]
        if not all(bbox.frame == ref.frame for bbox in bboxes):
            raise ValueError("Not all Provided BBoxes are from the same frame")

        return cls(
            frame_obj=ref.frame_obj,
            item_obj=item,
            xmin=min(i.xmin for i in bboxes),
            xmax=max(i.xmax for i in bboxes),
            ymin=min(i.ymin for i in bboxes),
            ymax=max(i.ymax for i in bboxes),
            conf=sum(i.conf for i in bboxes) / len(bboxes),
        )

    @property
    def item(self) -> str:
        return self.item_obj.key

    @property
    def type(self) -> ItemType:
        return self.item_obj.type

    @property
    def ignore(self) -> bool:
        return self.item_obj.ignore

    @property
    def area(self) -> float:
        """The total area of the bounding box."""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    @property
    def x(self) -> float:
        return (self.xmax + self.xmin) / 2

    @property
    def y(self) -> float:
        return (self.ymax + self.ymin) / 2

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    # @property
    def has_item(self) -> bool:
        return self.item_obj is not None

    def overlap_with(self, other: BBox) -> float:
        dx = min(self.xmax, other.xmax) - max(self.xmin, other.xmin)
        dy = min(self.ymax, other.ymax) - max(self.ymin, other.ymin)
        if (dx >= 0) and (dy >= 0):
            if self.item is None:
                raise ValueError("BBox is not assigned to an Item")
            if other.item is None:
                raise ValueError("other BBox is not assigned to an Item")
            # get overlapping area
            overlap = dx * dy
            return max((overlap / box.area) for box in (self, other))
        return 0

    def touching_boundary(self, w: int, h: int, buffer: int) -> bool:
        return self.xmin < buffer or self.xmax > w - buffer or self.ymin < buffer or self.ymax > h - buffer

    def distance_to(self, other: BBox) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def within(self, other: BBox) -> bool:
        return other.xmin <= self.x <= other.xmax and other.ymin <= self.y <= other.ymax


@dataclass
class Item:
    type: ItemType
    id: int
    sub_id: int = 0
    subject: Subject = field(default=Subject.NOT_CLASSIFIED, init=False)
    ignore: bool = False
    _id_distributor: IDDistributor = field(default_factory=IDDistributor)
    _boxes: FrameIndexer[BBox] = field(default_factory=FrameIndexer, init=False)
    _cache: FrameCache[BBox] | None = field(default=None, init=False)

    def __lt__(self, other: Item):
        if self.id < other.id:
            return True
        if self.id == other.id:
            return self.sub_id < other.sub_id
        return False

    def __le__(self, other: Item):
        if self.id < other.id:
            return True
        if self.id == other.id:
            return self.sub_id <= other.sub_id
        return False

    def __gt__(self, other: Item):
        if self.id > other.id:
            return True
        if self.id == other.id:
            return self.sub_id > other.sub_id
        return False

    def __ge__(self, other: Item):
        if self.id > other.id:
            return True
        if self.id == other.id:
            return self.sub_id >= other.sub_id
        return False

    def __eq__(self, other: Item):
        return self.type == other.type and self.id == other.id and self.sub_id == other.sub_id

    def __hash__(self) -> int:
        return hash((type(self), self.type, self.id, self.sub_id))

    def __enter__(self):
        self.start_caching()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.release_cache()

    @classmethod
    def from_str(cls, key: str) -> Self:
        _, item_type, ids = key.split("-")
        main_id, sub_id = (int(id_) for id_ in ids.split("."))
        return cls(type=ItemType(item_type), id=main_id, sub_id=sub_id)

    @classmethod
    def combine(cls, items: Sequence[Self]) -> Self:
        new = items[0]._make_child()
        new._inherit_timeline(items)  # pylint: disable=[protected-access]
        for item in items:
            item.ignore = True
        return new

    @classmethod
    def spawn_event(cls, id_: int, source: list[Item]) -> Self:
        ref = source[0]
        if ref.subject is None:
            raise ValueError("Items can not have no subject")
        if not all(i.subject == ref.subject for i in source):
            raise ValueError("Not all Items share the same subject")
        new = cls(type=ItemType.EVENT, id=id_)
        new.subject = ref.subject
        new._inherit_timeline(source)
        return new

    @property
    def track_len(self) -> int:
        return self.end - self.start + 1

    def _inherit_timeline(self, items: Iterable[Item]):
        if not all(item.subject == self.subject for item in items):
            raise ValueError("Items must have the same subject")
        start = min(i.start for i in items)
        end = max(i.end for i in items)
        with self:
            for f in range(start, end + 1):
                source_boxes = [s.boxes[f] for s in items if s.boxes.has(f)]
                if len(source_boxes) > 0:
                    BBox.surround(self, source_boxes)

    def start_caching(self):
        if self._cache is None:
            self._cache = self._boxes.get_cache()

    def release_cache(self):
        if self._cache is not None:
            self._cache.release()
            self._cache = None

    def add_bbox(self, bbox: BBox):
        if bbox.frame in self._boxes:
            raise ValueError("Item already has a BBox at specified frame")
        target = self._cache or self._boxes
        # target = self._boxes
        target.include(bbox)

    def destroy(self):
        self.ignore = True
        for box in self._boxes.get_all():
            box.frame_obj.bboxes.remove(box)

    @property
    def boxes(self) -> FrameIndexer[BBox]:
        return self._boxes

    @property
    def key(self):
        return f"cam-{self.type}-{self.id}.{self.sub_id}"

    @property
    def start(self) -> int:
        return self._boxes.start

    @property
    def end(self) -> int:
        return self._boxes.end

    def frame_overlap(self, other: Item) -> int:
        return max(min(self.end, other.end) - max(self.start, other.start), 0)

    def is_ring(self) -> bool:
        return self.type == ItemType.RING_METAL or self.type == ItemType.RING_PLASTIC

    def cut_at(self, frame: int) -> Self:
        new = self._make_child()
        new._boxes = self._boxes.cut(frame)  # pylint: disable=protected-access
        for box in new.boxes[:]:
            box.item_obj = new
        return new

    def _make_child(self) -> Self:
        child = replace(self, sub_id=self._id_distributor.get_id())
        child.subject = self.subject
        return child

    def get_gaps(self, size: int) -> Iterable[tuple[BBox, BBox]]:
        return [
            (self._boxes[start], self._boxes[stop]) for start, stop in self._boxes.get_frame_steps() if (stop - start) - 1 >= size
        ]


@dataclass
class Frame:
    frame: int
    _scene: Scene
    # frame_shape: tuple[int, int] = (1080, 1920)
    bboxes: ItemTypedCollection = field(default_factory=ItemTypedCollection, init=False)

    def __lt__(self, other: Frame):
        return self.frame < other.frame

    def __le__(self, other: Frame):
        return self.frame <= other.frame

    def __gt__(self, other: Frame):
        return self.frame > other.frame

    def __ge__(self, other: Frame):
        return self.frame >= other.frame

    def __eq__(self, other: Frame):
        return self.frame == other.frame

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self._scene.frame_shape

    @property
    def width(self) -> int:
        return self._scene.frame_shape[1]

    @property
    def height(self) -> int:
        return self._scene.frame_shape[0]

    @property
    def birds(self) -> Generator[BBox]:
        return self.bboxes.get(ItemType.BIRD)

    @property
    def rings(self) -> Generator[BBox]:
        return self.bboxes.get(ItemType.RING_METAL, ItemType.RING_PLASTIC)

    @property
    def mud(self) -> Generator[BBox]:
        return self.bboxes.get(ItemType.MUD)

    @property
    def orphans(self) -> Generator[BBox]:
        return self.bboxes.get(ignored=True)

    @property
    def events(self) -> Generator[BBox]:
        return self.bboxes.get(ItemType.EVENT)

    def has_rings(self) -> bool:
        return any(True for _ in self.bboxes.get(ItemType.RING_METAL, ItemType.RING_PLASTIC))

    def add_bbox(self, bbox: BBox):
        self.bboxes.add(bbox)
