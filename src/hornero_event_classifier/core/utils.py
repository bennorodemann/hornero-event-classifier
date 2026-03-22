from __future__ import annotations

import re
from collections import defaultdict
from threading import Event
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Self,
    TypedDict,
    overload,
)

import numpy as np
from sortedcontainers import SortedDict  # pylint: disable=import-error

from hornero_event_classifier.core.enums import ItemType

if TYPE_CHECKING:
    from hornero_event_classifier.core.data import Item
_id_pattern: re.Pattern = re.compile(r"^[a-zA-Z]*\d+(_[a-zA-Z]*\d+)+")


def extract_file_id(filename: str):
    out: re.Match | None = _id_pattern.match(filename)
    if out is None:
        return None
    else:
        return out.group()


class Comparable(Protocol):
    def __lt__(self, other: Self) -> bool: ...
    def __le__(self, other: Self) -> bool: ...
    def __gt__(self, other: Self) -> bool: ...
    def __ge__(self, other: Self) -> bool: ...
    def __eq__(self, other: Self) -> bool: ...


def compare_seq[T: HasFrame](data: FrameIndexer[T]) -> Iterator[tuple[T, T]]:
    sorted_objs = data._data.values()
    return zip(sorted_objs, sorted_objs[1:])


class DefaultSpawnDict[K, V](dict[K, V]):
    def __init__(
        self,
        obj_factory: Callable[..., V],
        iterable: Iterable[tuple[K, V]] = (),
        target_var: Optional[str] = None,
        defaults: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(iterable)
        self._obj_factory: Callable[..., V] = obj_factory
        self.target_var: str | None = target_var
        self.defaults: dict[str, Any] = defaults or {}

    def __missing__(self, key: K) -> V:
        if self.target_var:
            var_dict = {self.target_var: key}
            new = self._obj_factory(**var_dict, **self.defaults)
        else:
            new = self._obj_factory(key, **self.defaults)
        self[key] = new
        return new


class YOLOData(TypedDict):
    Frame: int
    Cam: str
    ID: str
    Xmin: float
    Ymin: float
    Xmax: float
    Ymax: float
    Conf: float


def type_yolo_data(data: dict[str, str]) -> YOLOData:
    return {
        "Frame": int(data["Frame"]),
        "Cam": data["Cam"],
        "ID": data["ID"],
        "Xmin": float(data["Xmin"]),
        "Ymin": float(data["Ymin"]),
        "Xmax": float(data["Xmax"]),
        "Ymax": float(data["Ymax"]),
        "Conf": float(data["Conf"]),
    }


class HasFrame(Protocol):
    @property
    def frame(self) -> int: ...


class FrameCache[T: HasFrame]:
    def __init__(self, parent: FrameIndexer) -> None:
        self._parent = parent
        self._cache = []

    def include(self, data: T):
        self._cache.append(data)

    def include_many(self, data: Iterable[T]):
        self._cache.extend(data)

    def release(self):
        self._parent.include_many(self._cache)
        self._cache = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.release()


class FrameIndexer[T: HasFrame]:
    def __init__(self, *args, **kwargs):
        self.start: int = 0
        self.end: int = 0
        self._data: SortedDict = SortedDict(*args, **kwargs)
        if self._data:
            self._refresh_range()

    def __repr__(self) -> str:
        return f"FrameIndexer({self.start} -> {self.end})"

    def _refresh_range(self) -> None:
        self.start = min(self._data)
        self.end = max(self._data)

    def include(self, data: T):
        self._data[data.frame] = data
        self._refresh_range()

    def include_many(self, data: Iterable[T]):
        self._data.update([(d.frame, d) for d in data])
        self._refresh_range()

    def get_frame_steps(self) -> Iterable[tuple[int, int]]:
        return ((k1, k2) for k1, k2 in zip(self._data.keys(), self._data.keys()[1:]))

    def get_all(self) -> Iterable[T]:
        return self._data.values()

    @classmethod
    def _spawn_with_items(cls, data: list[tuple[int, T]] | tuple[int, T]) -> Self:
        new = cls()
        new._data.update(data)
        new._refresh_range()
        return new

    def cut(self, frame: int) -> Self:
        index = self._data.bisect_left(frame)
        items = self._data.items()
        self._data = SortedDict(items[:index])
        self._refresh_range()
        return type(self)._spawn_with_items(items[index:])

    def has(self, frame: int) -> bool:
        return frame in self._data

    def is_continues(self) -> bool:
        return all(d == 1 for d in np.diff(self._data.keys()))

    def get_missing(self) -> list[int]:
        return [v for v in range(self.start, self.end + 1) if v not in self._data]

    def validate(self):
        assert all(k == v.frame for k, v in self._data.items())

    def get[G](self, frame: int, default: G) -> T | G:
        return self._data.get(frame, default)

    def get_cache(self) -> FrameCache[T]:
        return FrameCache(self)

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> list[T]: ...

    def __getitem__(self, key: int | slice) -> T | list[T]:
        if isinstance(key, slice):
            start = key.start
            stop = key.stop

            start_index = self._data.bisect_left(start if start is not None else float("-inf"))
            stop_index = self._data.bisect_right(stop if stop is not None else float("inf"))

            return self._data.values()[start_index:stop_index]
        return self._data[key]

    def __setitem__(self, key, value):
        if key != value.frame:
            raise ValueError("Provided key does not match values frame")
        self.include(value)

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)

    def __len__(self):
        return (self.end - self.start) + 1

    # @property
    # def start(self):
    #     return min(self._data.keys())

    # @property
    # def end(self):
    #     return max(self._data.keys())


class IDDistributor:
    def __init__(self) -> None:
        self._id = 0

    def __repr__(self) -> str:
        return f"IDDistributor({self._id})"

    def get_id(self) -> int:
        self._id += 1
        return self._id


class ItemTyped(Protocol):
    type: ItemType
    ignore: bool


class ItemTypedCollection[T: ItemTyped]:
    def __init__(self, starting_data: Iterable[T] = ()) -> None:
        self._data: dict[ItemType, set[T]] = defaultdict(set)
        for obj in starting_data:
            self.add(obj)

    @staticmethod
    def _no_filter(_: T) -> bool:
        return True

    @staticmethod
    def _include_ignore(obj: T) -> bool:
        return obj.ignore is True

    @staticmethod
    def _no_ignore(obj: T) -> bool:
        return obj.ignore is False

    _filter_funcs: dict[bool | None, Callable[[T], bool]] = {
        True: _include_ignore,
        False: _no_ignore,
        None: _no_filter,
    }

    def add(self, obj: T) -> None:
        self._data[obj.type].add(obj)

    def remove(self, obj: T) -> None:
        self._data[obj.type].discard(obj)

    def extend(self, objects: Iterable[T]) -> None:
        for obj in objects:
            self.add(obj)

    def get(self, *item_types: ItemType, ignored: bool | None = False) -> Generator[T]:
        filter_func = self._filter_funcs[ignored]
        if len(item_types) == 0:
            item_types = tuple(ItemType)
            # return (object for type_group in self._data.values() for object in type_group if filter_func(object))
        return (obj for item_type in item_types for obj in self._data[item_type] if filter_func(obj))

    def has(self, obj: T) -> bool:
        return obj in self._data[obj.type]

    def __iter__(self):
        return self.get()


class ResultDict(TypedDict):
    video_id: str
    subject: str
    start_frame: int
    end_frame: int
    mud: bool
