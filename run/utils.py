from pathlib import Path
from hornero_event_classifier import ThresholdClassifier, Classifier, Metric, pipelines, CONFIG
import json
import os

def load_default_classifier() -> Classifier:
    dir = Path(__file__).parent
    with open(dir/"weights.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    weights = {Metric[k]:v for k,v in data["weights"].items()}
    return ThresholdClassifier.from_dict(weights, data["threshold"])


def open_vid(video_id: str, frame: int): 
    for file in os.listdir(CONFIG.yolo_path):
        if file.startswith(video_id):
            _, scene = pipelines.classify(CONFIG.yolo_path / file, load_default_classifier(), show_progress=False)
            pipelines.animate(scene, frame=frame, auto_play=False)
            return