"""Public tool helpers for validation, plotting, and animation."""

from hornero_event_classifier.tools.validate_events import validate_events
from hornero_event_classifier.tools.plot import event_plot, event_validation_plot
from hornero_event_classifier.tools.recommend_weights import classify_with_boris, recommend_weights
from hornero_event_classifier.tools.animate import Animator

__all__ = [
    "validate_events",
    "event_plot",
    "event_validation_plot",
    "classify_with_boris",
    "recommend_weights",
    "Animator",
]
