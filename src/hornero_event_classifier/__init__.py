"""The hornero_event_classifier library is a system that detects visitation events of ringed and unringed hornero to a nest."""

from hornero_event_classifier import tools
from hornero_event_classifier.animate.animate import Animator
from hornero_event_classifier.classifiers import Classifier, Metric, SegmentCollection, Sequence, ThresholdClassifier
from hornero_event_classifier.core import (
    BBox,
    Frame,
    Item,
    ItemType,
    Scene,
    filters,
    VideoMetadata,
    gen_metadata,
    read_metadata,
    write_metadata,
)

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
    "Animator",
    "tools",
    "VideoMetadata",
    "gen_metadata",
    "read_metadata",
    "write_metadata",
]
