from pathlib import Path
from hornero_event_classifier import VideoMetadata, read_metadata, Metric, ThresholdClassifier
import pandas as pd
import sys
from hornero_event_classifier import tools
import numpy as np
from classify import classify


def _bool_to_subject(val: bool) -> str:
    return "ring" if val else "no_ring"


def recommend_weights(
    metrics: list[Metric] | None, yolo: pd.DataFrame, boris: pd.DataFrame
) -> tuple[tuple[float, dict[Metric, float]], pd.DataFrame]:
    classified = tools.classify_with_boris(yolo=yolo, boris=boris)
    intercept, weights = tools.recommend_weights(classified, metrics)

    classified = classified.rename(columns={"subject": "real_subject"})
    classified["calc_subject"] = (np.sum(classified.loc[:, weights.index] * weights, axis=1) >= intercept).map(_bool_to_subject)

    classified["offset"] = (classified.loc[:, weights.index] * weights).sum(axis=1) - intercept
    return (
        float(intercept),
        {Metric[metric]: float(weight) for metric, weight in zip(weights.index, weights.values)},
    ), classified


if __name__ == "__main__":
    CACHE_PATH = Path("data/segment_cache.csv")
    TEST_METRICS: list[Metric] | None = None
    refresh = len(sys.argv) > 1 and sys.argv[1] in ("reload", "refresh")

    metadata_repo: dict[str, VideoMetadata] = read_metadata("data/video_metadata.json")
    segment_dfs: list[pd.DataFrame] = []
    if refresh or CACHE_PATH is None or not CACHE_PATH.exists():
        for video_metadata in metadata_repo.values():
            classifier = ThresholdClassifier(list(Metric), [1 for _ in Metric])
            _, scene = classify(video_metadata, classifier)
            if scene.segments is not None:
                segment_dfs.append(scene.segments.as_df(video_metadata.name))
        segment_data: pd.DataFrame = pd.concat(segment_dfs)
        if CACHE_PATH is not None:
            segment_data.to_csv(CACHE_PATH)
    else:
        segment_data = pd.read_csv(CACHE_PATH)
    boris = pd.read_csv("data/DB_BORIS.csv")

    (threshold, weights), results = recommend_weights(TEST_METRICS, segment_data, boris)

    weights = {k.name: v for k, v in weights.items()}
    print(f"weights: {weights}")
    print(f"threshold: {threshold}")
    print(results[[col for col in results.columns]][results["real_subject"] != results["calc_subject"]])
    print(f"accuracy: {(results["real_subject"] == results["calc_subject"]).mean()}")
