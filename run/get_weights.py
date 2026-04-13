import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from classify import classify
from defaults import SEGMENTS_CACHE_PATH, METADATA_FILE, BORIS_FILE

from hornero_event_classifier import (
    Metric,
    ThresholdClassifier,
    VideoMetadata,
    read_metadata,
    tools,
)

# FIXME: needs more documentation


def _bool_to_subject(val: bool) -> str:
    return "ring" if val else "no_ring"


def recommend_weights(
    metrics: list[Metric] | None, yolo: pd.DataFrame, boris: pd.DataFrame
) -> tuple[tuple[float, dict[Metric, float]], pd.DataFrame]:
    classified = tools.classify_with_boris(yolo=yolo, boris=boris)
    intercept, weights = tools.recommend_weights(classified, metrics)

    classified = classified.rename(columns={"subject": "real_subject"})
    classified["calc_subject"] = (np.sum(classified.loc[:, weights.index] * weights, axis=1) >= intercept).map(
        _bool_to_subject
    )

    classified["offset"] = (classified.loc[:, weights.index] * weights).sum(axis=1) - intercept
    return (
        float(intercept),
        {Metric[metric]: float(weight) for metric, weight in zip(weights.index, weights.values)},
    ), classified


parser = ArgumentParser()
parser.add_argument("metrics", choices=list(Metric), type=lambda m: Metric[m], nargs="*")
parser.add_argument("--refresh", action="store_true", help="Recalculate segment metrics")

if __name__ == "__main__":
    args = parser.parse_args()

    metadata_repo: dict[str, VideoMetadata] = read_metadata(METADATA_FILE)
    segment_dfs: list[pd.DataFrame] = []
    # if requested, run classify pipeline and pull segment data
    if args.refresh or SEGMENTS_CACHE_PATH is None or not SEGMENTS_CACHE_PATH.exists():
        for video_metadata in metadata_repo.values():
            classifier = ThresholdClassifier(list(Metric), [1 for _ in Metric])
            _, scene = classify(video_metadata, classifier)
            if scene.segments is not None:
                segment_dfs.append(scene.segments.as_df(video_metadata.name))
        segment_data: pd.DataFrame = pd.concat(segment_dfs)
        if SEGMENTS_CACHE_PATH is not None:
            segment_data.to_csv(SEGMENTS_CACHE_PATH)
    else:
        segment_data = pd.read_csv(SEGMENTS_CACHE_PATH)
    boris = pd.read_csv(BORIS_FILE)

    (threshold, weights), results = recommend_weights(args.metrics, segment_data, boris)

    # Print results
    print(f"weights: {weights}")
    print(f"threshold: {threshold}")
    print(results[[col for col in results.columns]][results["real_subject"] != results["calc_subject"]])
    print(f"accuracy: {(results["real_subject"] == results["calc_subject"]).mean()}")
