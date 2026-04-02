import json
from pathlib import Path

import pipelines

from hornero_event_classifier import (
    Classifier,
    Metric,
    ThresholdClassifier,
    VideoMetadata,
)


def load_default_classifier() -> Classifier:
    dir_ = Path(__file__).parent
    with open(dir_ / "weights.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    weights = {Metric[k]: v for k, v in data["weights"].items()}
    return ThresholdClassifier.from_dict(weights, data["threshold"])


def open_vid(video_metadata: VideoMetadata, frame: int):
    if not video_metadata.video_path.exists():
        print(f"Video {video_metadata.name} not found at: {video_metadata.video_path}")
        return
    _, scene = pipelines.classify(video_metadata, load_default_classifier(), show_progress=False)
    pipelines.animate(scene, scale=2, frame=frame, auto_play=False)
    return
