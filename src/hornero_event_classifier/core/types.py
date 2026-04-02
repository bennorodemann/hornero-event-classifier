"""Shared type protocols and typed dictionaries for core data handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, TypedDict

if TYPE_CHECKING:
    from hornero_event_classifier.core.enums import ItemType


class Comparable(Protocol):
    """
    A helper :code:`Protocol` class describing a class that supports :code:`<`, :code:`<=`, :code:`>`, :code:`>=` and :code:`==`
    """

    def __lt__(self, other: Self) -> bool: ...
    def __le__(self, other: Self) -> bool: ...
    def __gt__(self, other: Self) -> bool: ...
    def __ge__(self, other: Self) -> bool: ...
    def __eq__(self, other: Self) -> bool: ...


class YOLOData(TypedDict):
    """A :code:`TypedDict` describing raw YOLO csv inputs. For static type checking purposes."""

    #: frame number
    Frame: int
    #: cam string
    Cam: str
    #: bbox ID
    ID: str
    #: bbox minimum x value
    Xmin: float
    #: bbox minimum y value
    Ymin: float
    #: bbox maximum x value
    Xmax: float
    #: bbox maximum y value
    Ymax: float
    #: bbox confidence
    Conf: float


def type_yolo_data(data: dict[str, str]) -> YOLOData:
    """Turns a dict of strings from a YOLO csv row into a typed dict following :py:class:`YOLOData`.

    :param data: YOLO csv row dict
    :type data: dict[str, str]
    :return: typed dict, following :py:class:`YOLOData`
    :rtype: YOLOData
    """
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


class ResultDict(TypedDict):
    """A :code:`TypedDict` describing a output result of a classification."""

    #: source video id
    video_id: str
    #: if event refers to ringed or unringed bird
    subject: str
    #: the frame where the event starts
    start_frame: int
    #: the frame where the event ends
    end_frame: int
    #: if the bird had mud (currently always :code:`False`)
    mud: bool


class HasFrame(Protocol):
    """A helper :code:`Protocol` class describing a class with a :code:`frame` attribute of type :code:`int`"""

    @property
    def frame(self) -> int:  # type: ignore
        """an integer referring to the object's frame

        :rtype: int
        """


class ItemTyped(Protocol):
    """A :code:`Protocol` describing a class with have a :py:class:`ItemType` and can have an ignored state."""

    #: the type of the item
    type: ItemType
    #: if the instance should be ignored
    ignore: bool
