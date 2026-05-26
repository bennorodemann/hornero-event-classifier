"""Core package for detecting visitation events of ringed and unringed hornero.

This package aggregates the public API used in internal workflows.
"""

from hornero_event_classifier.classifiers.base import Classifier, SegmentCollection, Sequence
from hornero_event_classifier.classifiers.metrics import Metric
from hornero_event_classifier.classifiers.threshold import ThresholdClassifier
from hornero_event_classifier.core import filters
from hornero_event_classifier.core.data import BBox, Frame, Item
from hornero_event_classifier.core.enums import ItemType
from hornero_event_classifier.core.scene import Scene
from hornero_event_classifier.core.video_metadata import VideoMetadata, gen_metadata, read_metadata, write_metadata

__all__ = [
    "Scene",
    "filters",
    "ItemType",
    "Metric",
    "ThresholdClassifier",
    "Classifier",
    "Sequence",
    "SegmentCollection",
    "BBox",
    "Frame",
    "Item",
    "VideoMetadata",
    "gen_metadata",
    "read_metadata",
    "write_metadata",
]


def __getattr__(name: str):
    if name == "tools":
        from hornero_event_classifier import tools

        return tools
    raise AttributeError(name)
