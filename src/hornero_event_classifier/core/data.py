from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Generator,
    Iterable,
    Self,
    Sequence,
)
from itertools import count

from hornero_event_classifier.core.enums import ItemType, Subject
from hornero_event_classifier.core.utils import (
    FrameCache,
    FrameIndexer,
    ItemTypedCollection,
    YOLOData,
)

if TYPE_CHECKING:
    from hornero_event_classifier.core.video_data import VideoMetadata


@dataclass
class BBox:
    """Bounding box ``dataclass`` for a specific frame with references to corresponding :py:class:`Frame` and :py:class:`Item`.

    This class is comparable using ``<``, ``<=``, ``>``, ``>=`` and ``==`` all of which compare :py:attr:`BBox.frame` of both
    :py:class:`BBox`\\s. It is also hashable.

    :param frame_obj: Related frame object
    :type frame_obj: Frame
    :param item_obj: Related item object
    :type item_obj: Item
    :param xmin: Minimum x value (from top of frame)
    :type xmin: float
    :param xmax: Maximum x value (from top of frame)
    :type xmax: float
    :param ymin: Minimum y value (from left of frame)
    :type ymin: float
    :param ymax: Minimum y value (from left of frame)
    :type ymax: float
    :param conf: YOLOs confidence in bounding box
    :type conf: float
    :param real: True if source is from YOLO, False if created by :py:mod:`hornero_event_classifier`
    :type real: bool
    """

    frame_obj: Frame
    item_obj: Item
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    conf: float
    real: bool = True
    #: a cache where relevant info can be stored, primarily used by :py:class:`.Classifier`\\s
    metrics_cache: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
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
        """Spawn a new instance from a YOLO csv row dictionary.

        :param data: a row from a YOLO csv as a ``dict``
        :type data: YOLOData
        :param item: :py:class:`Item` to pass as :py:attr:`BBox.item_obj` argument.
        :type item: Item
        :param frame: :py:class:`Frame` to pass as :py:attr:`BBox.frame_obj` argument.
        :type frame: Frame
        :return: A new instance.
        :rtype: Self
        """
        return cls(frame, item, data["Xmin"], data["Xmax"], data["Ymin"], data["Ymax"], data["Conf"])

    @classmethod
    def surround(cls, item: Item, bboxes: list[BBox] | tuple[BBox, ...]) -> Self:
        """Create a new instance that surrounds multiple other :py:class:`BBox`\\s that are all in the same frame.

        :param item: :py:class:`Item` to attach instance to.
        :type item: Item
        :param bboxes: ``BBox``\\s to surround.
        :type bboxes: list[BBox] | tuple[BBox, ...]
        :raises ValueError: Raises an error not all ``BBox``\\s from ``bboxes`` are from the same frame.
        :return: A new instance.
        :rtype: Self
        """
        ref = bboxes[0]
        # check all bboxes are in the same frame
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
            real=False,
        )

    @property
    def item(self) -> str:
        """Shortcut for :py:attr:`BBox.item_obj.key <Item.key>`."""
        return self.item_obj.key

    @property
    def type(self) -> ItemType:
        """Shortcut for :py:attr:`BBox.item_obj.type <Item.type>`."""
        return self.item_obj.type

    @property
    def ignore(self) -> bool:
        """Shortcut for :py:attr:`BBox.item_obj.ignore <Item.ignore>`."""
        return self.item_obj.ignore

    @property
    def frame(self) -> int:
        """Shortcut for :py:attr:`BBox.frame_obj.frame <Frame.frame>`."""
        return self.frame_obj.frame

    @property
    def x(self) -> float:
        """The center x position of bounding box."""
        return (self.xmax + self.xmin) / 2

    @property
    def y(self) -> float:
        """The center y position of bounding box."""
        return (self.ymax + self.ymin) / 2

    @property
    def width(self) -> float:
        """The width of the bounding box."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """The height of the bounding box."""
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """The total area of the bounding box."""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def overlap_with(self, other: BBox) -> tuple[float, float]:
        """Get the percent overlap of two ``BBox``\\es

        :param other: Another ``BBox`` to compare against
        :type other: BBox
        :return: Two ``float``\\s, the first representing the overlap with the current ``BBox``, and the second the overlap with
            ``other``
        :rtype: tuple[float, float]
        """
        # get x overlap
        dx = min(self.xmax, other.xmax) - max(self.xmin, other.xmin)
        # get y overlap
        dy = min(self.ymax, other.ymax) - max(self.ymin, other.ymin)
        if (dx >= 0) and (dy >= 0):
            # get overlapping area
            overlap = dx * dy
            return (overlap / self.area, overlap / other.area)
        return (0, 0)

    def touching_boundary(self, buffer: int) -> bool:
        """Check if the ``BBox`` is within ``buffer`` frames of the frame boundary.

        :param buffer: number of pixels from the boundary to accept as "touching".
        :type buffer: int
        :return: ``True`` if any of the boundaries are within ``buffer`` pixel of the frame border. ``False`` otherwise.
        :rtype: bool
        """
        h, w = self.frame_obj.frame_shape
        return self.xmin < buffer or self.xmax > w - buffer or self.ymin < buffer or self.ymax > h - buffer

    def distance_to(self, other: BBox) -> float:
        """Distance (in pixels) between the center point of two ``BBox``\\s"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def within(self, other: BBox) -> bool:
        """Check if the center point of ``other`` is within the boundaries of the current ``BBox``

        :param other: other ``BBox`` to compare with
        :type other: BBox
        :return: Returns ``True`` if center point of ``other`` is within current ``BBox``. ``False`` otherwise.
        :rtype: bool
        """
        return other.xmin <= self.x <= other.xmax and other.ymin <= self.y <= other.ymax


@dataclass
class Item:
    """A grouping of related :py:class:`BBox`\\s of a specific :py:class:`.ItemType`.

    This class is comparable using ``<``, ``<=``, ``>``, ``>=`` and ``==`` all of which first compare the ``id`` of both items. If
    both are equal ``sub_id`` is compared. This class is also hashable.

    This class can be entered with the ``with`` command to temporarily cache added :py:class:`BBox`\\s for performance gains (see
    :py:class:`.FrameCache`). Upon leaving the with statement the cache is automatically released if no errors were raised. This
    is the same as calling :py:meth:`Item.start_caching` and then :py:meth:`Item.release_cache`.

    .. code-block:: python3

        item = Item(...)
        with item:
            # add BBoxes
        # do stuff with BBoxes

    :param type: What type of item this instance refers to.
    :type type: ItemType
    :param id: The parent ID, generally inherited from YOLO (except for ``Item``\\s of type: :py:attr:`.ItemType.EVENT`).
    :type id: int
    :param sub_id: A secondary ID which is incremented whenever a ``Item`` is :py:meth:`cut <Item.cut_at>`. Default is ``0``.
    :type sub_id: int
    :param subject: In the case of ``Item``\\s of type :py:attr:`.ItemType.BIRD` this is where classifications are stored.
        Otherwise this attribute is ignored. Default is :py:attr:`.Subject.NOT_CLASSIFIED`.
    :type subject: Subject
    :param ignore: A attribute that indicates if the ``Item`` should be ignored in future calculations.
    :type ignore: bool"""

    type: ItemType
    id: int
    sub_id: int = 0
    subject: Subject = field(default=Subject.NOT_CLASSIFIED, init=False)
    ignore: bool = False
    _id_counter: count = field(default_factory=lambda: count(1))
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

    @property
    def boxes(self) -> FrameIndexer[BBox]:
        return self._boxes

    @property
    def key(self) -> str:
        """A reference ``str`` with pattern ``cam-<type>-<id>.<sub_id>``."""
        return f"cam-{self.type}-{self.id}.{self.sub_id}"

    @property
    def start(self) -> int:
        return self._boxes.start

    @property
    def end(self) -> int:
        return self._boxes.end

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

    def _make_child(self) -> Self:
        child = replace(self, sub_id=next(self._id_counter))
        child.subject = self.subject
        return child

    def start_caching(self):
        """Start caching added :py:class:`BBox`\\s.

        After calling this method, any added :py:class:`BBox`\\s using :py:meth:`Item.add_bbox` will first be cached and will
        not be normally accessible until :py:meth:`Item.release_cache` is called. This helps with performance when adding many
        :py:class:`BBox`\\s (see :py:class:`.FrameCache`).
        """
        if self._cache is None:
            self._cache = self._boxes.get_cache()

    def release_cache(self):
        """Release cache and add all cached items to ``Item``"""
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

    def get_gaps(self, size: int) -> Iterable[tuple[BBox, BBox]]:
        return [
            (self._boxes[start], self._boxes[stop]) for start, stop in self._boxes.get_frame_steps() if (stop - start) - 1 >= size
        ]


@dataclass
class Frame:
    frame: int
    video_metadata: VideoMetadata
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
        return (self.video_metadata.height, self.video_metadata.width)

    @property
    def width(self) -> int:
        return self.video_metadata.width

    @property
    def height(self) -> int:
        return self.video_metadata.height

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
