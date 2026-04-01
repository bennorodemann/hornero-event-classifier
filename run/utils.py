from pathlib import Path
from hornero_event_classifier import ThresholdClassifier, Classifier, Metric, tools
import json

import pipelines


def load_default_classifier() -> Classifier:
    dir = Path(__file__).parent
    with open(dir / "weights.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    weights = {Metric[k]: v for k, v in data["weights"].items()}
    return ThresholdClassifier.from_dict(weights, data["threshold"])


def open_vid(video_metadata: tools.VideoMetadata, frame: int):
    _, scene = pipelines.classify(video_metadata, load_default_classifier(), show_progress=False)
    pipelines.animate(scene, scale=2, frame=frame, auto_play=False)
    return
