from pathlib import Path
from hornero_event_classifier import CONFIG, pipelines
from hornero_event_classifier.classifiers import Metric, ThresholdClassifier
import os
import pandas as pd
import sys

CACHE_PATH = Path(CONFIG.data_root / "segment_cache.csv")
TEST_METRICS: list[Metric] | None = None
refresh = len(sys.argv) > 1 and sys.argv[1] in ("reload", "refresh")

segment_dfs: list[pd.DataFrame] = []
if refresh or CACHE_PATH is None or not CACHE_PATH.exists():
    for file in os.listdir(CONFIG.yolo_path):
        classifier = ThresholdClassifier(list(Metric), [1 for _ in Metric])
        _, scene = pipelines.classify(CONFIG.yolo_path / file, classifier)
        if scene.segments is not None:
            segment_dfs.append(scene.segments.as_df(scene.video_id))
    segment_data: pd.DataFrame = pd.concat(segment_dfs)
    if CACHE_PATH is not None:
        segment_data.to_csv(CACHE_PATH)
else:
    segment_data = pd.read_csv(CACHE_PATH)
boris = pd.read_csv(CONFIG.boris_path)

(threshold, weights), results = pipelines.recommend_weights(TEST_METRICS, segment_data, boris)

weights = {k.name:v for k,v in weights.items()}
print(f"weights: {weights}")
print(f"threshold: {threshold}")
print(results[[col for col in results.columns]][results["real_subject"] != results["calc_subject"]])
print(f"accuracy: {(results["real_subject"] == results["calc_subject"]).mean()}")
