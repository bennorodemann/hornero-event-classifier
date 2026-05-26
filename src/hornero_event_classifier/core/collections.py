"""Collection and indexing helpers used across core data structures."""

from __future__ import annotations

from collections import defaultdict
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Iterator,
    Optional,
    TypeVar,
    overload,
)

import numpy as np
from sortedcontainers import SortedDict  # pylint: disable=import-error

from hornero_event_classifier.core.enums import ItemType
from hornero_event_classifier.core.types import HasFrame, ItemTyped

try:
    from typing import Self
except ImportError:  # pragma: no cover - Python < 3.11
    from typing_extensions import Self

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T", bound=HasFrame)
TItem = TypeVar("TItem", bound=ItemTyped)
G = TypeVar("G")

class DefaultSpawnDict(dict[K, V], Generic[K, V]):
    """A custom dictionary subclass similar to :py:class:`collections.defaultdict`. This class works like a dictionary except
    that if a value is indexed and does not exist, then a new object is created using the index key as an input argument.

    :param obj_factory: a callable that takes at least one input argument and returns an object
    :type obj_factory: Callable[..., V]
    :param iterable: starting dictionary mappings, defaults to ()
    :type iterable: Iterable[tuple[K, V]], optional
    :param target_var: key word argument in :code:`obj_factory` to pass missing keys. If None (the default), key is passed as
        first argument
    :type target_var: Optional[str], optional
    :param defaults: dict of default values that are also passed to :code:`obj_factory`, defaults to None
    :type defaults: Optional[dict[str, Any]], optional
    """

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


class FrameIndexer(Generic[T]):
    """A class organizing a sequence of objects along a timeline. In particular, this class allows for indexing objects based on
    their frame number. :py:class:`FrameIndexer` does not support holding multiple objects per frame. All objects in
    :py:class:`FrameIndexer` have a :code:`frame` variable of type :code:`int` which it will automatically sort.

    :py:class:`FrameIndexer` supports indexing using the frame number of the objects. If a slice is provided then all object in
    the slice are returned. If any frames are missing, those are not returned.

    .. code-block:: python3
        :linenos:
        :emphasize-lines: 16-18

        class MyClass:
            def __init__(self, frame: int):
                self.frame = frame

            def __str__(self) -> str:
                return f"frame: {self.frame}"

        indexer = FrameIndexer()
        indexer.include(MyClass(3))
        indexer.include_many([MyClass(4),MyClass(10)])

        indexer[3]
        >>> "frame: 3"
        indexer[3:20]
        >>> ["frame: 3", "frame: 4", "frame: 10"]
        # slices are end inclusive!!!
        indexer[3:4]
        >>> ["frame: 3", "frame: 4"]
        indexer[7] = MyClass(6)
        >>> ValueError
        indexer[6] = MyClass(6)
        >>> None
        len(indexer)
        >>> 4
        for v in indexer:
            print(v)
        >>> "frame: 3"
        >>> "frame: 4"
        >>> "frame: 6"
        >>> "frame: 10"

    .. warning:: Unlike most of python, :py:class:`FrameIndexer` slicing is end inclusive.

    .. note:: The length of a :py:class:`FrameIndexer` is the inclusive span from the minimum to maximum frame
        (:code:`end - start + 1`), not the count of stored items. If there are gaps, the length can be greater than the number
        of items.
    """

    def __init__(self, *args, **kwargs):
        self.start: int = 0
        self.end: int = 0
        self._data: SortedDict = SortedDict(*args, **kwargs)
        if self._data:
            self._refresh_range()

    def __repr__(self) -> str:
        return f"FrameIndexer({self.start} -> {self.end})"

    @overload
    def __getitem__(self, key: int) -> T: ...

    @overload
    def __getitem__(self, key: slice) -> list[T]: ...

    def __getitem__(self, key: int | slice) -> T | list[T]:
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            # find the indexes the contain the specified slice
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
        """Return the inclusive span from :py:attr:`start` to :py:attr:`end`, not the count of stored items."""
        return (self.end - self.start) + 1

    @classmethod
    def _spawn_with_items(cls, data: list[tuple[int, T]] | tuple[int, T]) -> Self:
        new = cls()
        new._data.update(data)
        new._refresh_range()
        return new

    def _refresh_range(self) -> None:
        self.start = min(self._data)
        self.end = max(self._data)

    def include(self, data: T):
        """Add an object to the :py:class:`FrameIndexer`.

        :param data: Object to add that follows the :py:class:`HasFrame` protocol
        :type data: T
        """
        self._data[data.frame] = data
        self._refresh_range()

    def include_many(self, data: Iterable[T]):
        """Add several objects to :py:class:`FrameIndexer`.

        :param data: Objects to add that follow the :py:class:`HasFrame` protocol
        :type data: Iterable[T]
        """
        self._data.update([(d.frame, d) for d in data])
        self._refresh_range()

    def get(self, frame: int, default: G) -> T | G:
        """Return the object at the specified frame. If the frame does not exist, ``default`` is returned.

        :param frame: frame number
        :type frame: int
        :param default: default value to return if no object is found
        :type default: Any
        :return: The object at the specified frame number if it exists. Otherwise, ``default`` is returned.
        :rtype: T | G
        """
        return self._data.get(frame, default)

    def get_frame_steps(self) -> Iterable[tuple[int, int]]:
        """Returns a :code:`iterable` of :code:`tuples` along the frame sequence representing the frame number of each item and
        the following item. (e.g. :code:`[(1, 2), (2, 3), (3, 10), (10, 11)]`)

        :return: a series of :code:`tuples` with the previous and following frame along the frame sequence
        :rtype: Iterable[tuple[int, int]]
        """
        return ((k1, k2) for k1, k2 in zip(self._data.keys(), self._data.keys()[1:]))

    def get_all(self) -> Iterable[T]:
        """Return all values in order.

        :return: all values in order
        :rtype: Iterable[T]
        """
        return self._data.values()

    def cut(self, frame: int) -> Self:
        """Splits the :py:class:`FrameIndexer` in two removing cut objects from current instance and returning a new instance with
        the cut objects. The cut frame does not need to be within the :py:class:`FrameIndexer`\\s frame range.

        :param frame: The frame to cut at. The specified frame is also cut and returned with the new instance.
        :type frame: int
        :return: A new :py:class:`FrameIndexer` instance with the specified cut frame and all following objects.
        :rtype: Self
        """
        index = self._data.bisect_left(frame)
        items = self._data.items()
        self._data = SortedDict(items[:index])
        self._refresh_range()
        return type(self)._spawn_with_items(items[index:])

    def has(self, frame: int) -> bool:
        """Check if :py:class:`FrameIndexer` contains an object at a specific frame.

        :param frame: The frame number to check
        :type frame: int
        :return: returns :code:`True` if there is an object at that frame number, otherwise returns :code:`False`
        :rtype: bool
        """
        return frame in self._data

    def is_continues(self) -> bool:
        """Check whether there are no missing frames between the minimum and maximum frames.

        :return: returns :code:`True` if there are no gaps. Otherwise return :code:`False`
        :rtype: bool
        """
        return all(d == 1 for d in np.diff(self._data.keys()))

    def get_missing(self) -> list[int]:
        """Get all missing frames between the minimum and maximum frames.

        :return: list of frame numbers which do not exist in the :py:class:`FrameIndexer` instance
        :rtype: list[int]
        """
        return [v for v in range(self.start, self.end + 1) if v not in self._data]

    def validate(self):
        """Double check that all frame number indexes correspond to the saved object.

        This is for debugging purposes and does not need to be used by users.
        """
        assert all(k == v.frame for k, v in self._data.items())

    def get_cache(self) -> FrameCache[T]:
        """Get a :py:class:`FrameCache` instance with current :py:class:`FrameIndexer` instance as parent.

        :return: :py:class:`FrameCache` instance with current :py:class:`FrameIndexer` instance as parent.
        :rtype: FrameCache[T]
        """
        return FrameCache(self)


class FrameCache(Generic[T]):
    """This class temporarily holds objects before passing them to a :py:class:`FrameIndexer`.

    This class helps with optimization as :py:class:`FrameIndexer` needs to sort everything every time an object is added.
    This class can be entered using the :code:`with` statement which at exit will automatically call
    :py:meth:`FrameCache.release` if no error occurred.

    :param parent: a :py:class:`FrameIndexer` instance to pass held objects to on release
    :type parent: FrameIndexer
    """

    def __init__(self, parent: FrameIndexer) -> None:
        self._parent = parent
        self._cache = []

    def include(self, data: T):
        """Add an object to the cache.

        :param data: Object to be added.
        :type data: T
        """
        self._cache.append(data)

    def include_many(self, data: Iterable[T]):
        """Add multiple objects to the cache.

        :param data: An iterable of objects to be added.
        :type data: Iterable[T]
        """
        self._cache.extend(data)

    def release(self):
        """Adds all cached items to parent :py:class:`FrameIndexer` and clears cache."""
        self._parent.include_many(self._cache)
        self._cache = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.release()


class ItemTypedCollection(Generic[TItem]):
    """A collection of items that follow the :py:class:`ItemTyped` protocol.

    :param starting_data: A collection of starting data, defaults to ()
    :type starting_data: Iterable[T], optional
    """

    def __init__(self, starting_data: Iterable[TItem] = ()) -> None:
        self._data: dict[ItemType, set[TItem]] = defaultdict(set)
        for obj in starting_data:
            self.add(obj)

    @staticmethod
    def _no_filter(_: TItem) -> bool:
        return True

    @staticmethod
    def _include_ignore(obj: TItem) -> bool:
        return obj.ignore is True

    @staticmethod
    def _no_ignore(obj: TItem) -> bool:
        return obj.ignore is False

    _filter_funcs: dict[bool | None, Callable[[TItem], bool]] = {
        True: _include_ignore,
        False: _no_ignore,
        None: _no_filter,
    }

    def add(self, obj: TItem) -> None:
        """Add an object to the collection.

        :param obj: Object to be added.
        :type obj: T
        """
        self._data[obj.type].add(obj)

    def extend(self, objects: Iterable[TItem]) -> None:
        """Add multiple object to the collection.

        :param objects: Iterable of objects to be added.
        :type objects: Iterable[T]
        """
        for obj in objects:
            self.add(obj)

    def remove(self, obj: TItem) -> None:
        """Remove an object from the collection.

        :param obj: Object to be removed.
        :type obj: T
        """
        self._data[obj.type].discard(obj)

    def get(self, *item_types: ItemType, ignored: bool | None = False) -> Generator[TItem, None, None]:
        """Return a generator of items of the specified types.

        If no ``item_types`` are provided, all types are included.

        :param item_types: One or more :py:class:`ItemType` values to include.
        :type item_types: ItemType
        :param ignored: If ``False`` (the default), only objects where ``obj.ignore is False`` are returned. If ``True``,
            only objects where ``obj.ignore is True`` are returned. If ``None``, both ignore and non-ignore objects are returned.
        :type ignored: bool | None, optional
        :return: Generator of objects following the filtering arguments.
        :rtype: Generator[T]
        """
        filter_func = self._filter_funcs[ignored]
        if len(item_types) == 0:
            item_types = tuple(ItemType)
        return (obj for item_type in item_types for obj in self._data[item_type] if filter_func(obj))

    def has(self, obj: TItem) -> bool:
        """Check if object is in the instance.

        :param obj: The object to check for.
        :type obj: T
        :return: A boolean representing if the object was found or not.
        :rtype: bool
        """
        return obj in self._data[obj.type]

    def __iter__(self):
        return self.get()
