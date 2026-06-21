"""
Weight recommendation utility for hornero event classification.

This module provides functionality to compute optimal linear classification
weights and thresholds based on labeled data. It combines YOLO-derived
segment features with BORIS ground-truth annotations to fit a simple
threshold-based classifier.

The main entry point is :func:`recommend_weights`, which:
- Aligns predicted features with annotated ground truth
- Computes metric weights and an intercept using helper utilities
- Applies the resulting model to evaluate classification performance

This script can also be run as a CLI tool to:
- Recompute or load cached segment-level features
- Fit weights for a selected subset of metrics
- Output misclassifications and overall accuracy
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from classify import classify, load_default_classifiers
from config import config
import json
from hornero_event_classifier import (
    Metric,
    ThresholdClassifier,
    VideoMetadata,
    read_metadata,
    tools,
)


def _bool_to_subject(val: bool) -> str:
    """
    Convert a boolean classification result into a subject label.

    :param val: Boolean value representing classification output.
    :type val: bool
    :return: "ring" if True, otherwise "no_ring".
    :rtype: str
    """
    return "ring" if val else "no_ring"


def recommend_weights(
    target: str,
    metrics: list[Metric] | None,
    yolo: pd.DataFrame,
    boris: pd.DataFrame,
) -> tuple[tuple[float, dict[Metric, float]], pd.DataFrame]:
    """
    Compute optimal weights and threshold for classification based on YOLO and BORIS data.

    This function merges YOLO predictions with BORIS annotations, computes
    recommended weights for the provided metrics, and applies the resulting
    linear model to classify subjects.

    :param target: name of classification metric
    :type target: str
    :param metrics: List of metrics to include in the weight calculation.
                    If None, all available metrics may be used.
    :type metrics: list[Metric] | None
    :param yolo: DataFrame containing YOLO-generated features.
    :type yolo: pandas.DataFrame
    :param boris: DataFrame containing BORIS ground-truth annotations.
    :type boris: pandas.DataFrame
    :return: A tuple containing:
             - A tuple with the intercept (threshold) and a dictionary of metric weights.
             - A DataFrame with classification results and additional computed columns.
    :rtype: tuple[tuple[float, dict[Metric, float]], pandas.DataFrame]
    """
    # Merge YOLO predictions with BORIS annotations
    classified = tools.classify_with_boris(target, yolo=yolo, boris=boris)

    if target == "subject":
        classified[target] = classified[target] == "ring"
    # Compute recommended weights and intercept
    intercept, weights = tools.recommend_weights(target, classified, metrics)

    # Rename ground-truth column for clarity
    classified = classified.rename(columns={target: "real_" + target})

    # Compute predicted subject using linear combination of metrics
    classified["calc_" + target] = np.sum(classified.loc[:, weights.index] * weights, axis=1) >= intercept
    if target == "subject":
        classified["calc_" + target] = classified["calc_" + target].map(_bool_to_subject)

    # Compute offset from decision boundary
    classified["offset"] = (classified.loc[:, weights.index] * weights).sum(axis=1) - intercept

    return (
        float(intercept),
        {Metric[metric]: float(weight) for metric, weight in zip(weights.index, weights.values)},
    ), classified


# Argument parser configuration
parser = ArgumentParser()
parser.add_argument("target", choices=["subject", "mud"], type=str)
parser.add_argument(
    "metrics",
    choices=list(Metric),
    type=lambda m: Metric[m],
    nargs="*",
    help="Optional list of metrics to use for weight calculation.",
)
parser.add_argument(
    "--refresh",
    action="store_true",
    help="Recalculate segment metrics instead of using cached data.",
)


if __name__ == "__main__":
    # Entry point for the script.

    # This script loads metadata, computes or retrieves segment data,
    # calculates optimal weights for classification, and outputs
    # performance statistics.

    args = parser.parse_args()

    # Load metadata repository
    metadata_repo: dict[str, VideoMetadata] = read_metadata(config.metadata_file)

    segment_dfs: list[pd.DataFrame] = []

    # if cache already exists, read the first line and mark refresh if first line does not match target
    if config.segments_cache_path.exists():
        with open(config.segments_cache_path, "r", encoding="utf-8") as file:
            if file.readline() != (args.target + "\n"):
                args.refresh = True

    # Generate or load segment data
    if args.refresh or config.segments_cache_path is None or not config.segments_cache_path.exists():
        for video_metadata in metadata_repo.values():
            if video_metadata.yolo_path.exists():
                if args.target == "subject":
                    # Initialize classifier with weights of 1 for so unedited metric values can be collected later
                    subject_classifier = ThresholdClassifier(list(Metric), [1 for _ in Metric])
                    mud_classifier = None
                else:
                    # load saved subject weights
                    subject_classifier = load_default_classifiers()["subject"]
                    # Initialize classifier with weights of 1 for so unedited metric values can be collected later
                    mud_classifier = ThresholdClassifier(list(Metric), [1 for _ in Metric])

                # Run classification pipeline
                _, scene = classify(video_metadata, subject_classifier, mud_classifier, remove_low_conf=0)

                # Collect segment data if available
                if scene.segments[args.target] is not None:
                    segment_dfs.append(scene.segments[args.target].as_df(video_metadata.name))
            else:
                print(f"yolo file for {video_metadata.name} not found ({video_metadata.file_path})")

        # Concatenate all segment data into a single DataFrame
        segment_data: pd.DataFrame = pd.concat(segment_dfs)

        # Cache segment data if a path is provided
        if config.segments_cache_path is not None:
            with open(config.segments_cache_path, "w", encoding="utf-8") as file:
                # write target of cache
                file.write(args.target + "\n")
                segment_data.to_csv(file)
    else:
        # Load cached segment data
        with open(config.segments_cache_path, "r", encoding="utf-8") as file:
            # skip over target
            file.readline()
            segment_data = pd.read_csv(file)

    # Load BORIS annotations
    boris = pd.read_csv(config.boris_file)

    # Compute weights and classification results
    (threshold, weights), results = recommend_weights(args.target, args.metrics, segment_data, boris)

    metric_strs = [metric.name for metric in args.metrics]
    simple_weights = {k.name: v for k, v in weights.items()}

    # Output results
    print(f"weights: {json.dumps(simple_weights)}")
    print(f"threshold: {threshold}")

    # Show misclassified rows
    print(
        results[[col for col in results.columns if (col in metric_strs or not metric_strs) or col.islower()]][
            results["real_" + args.target] != results["calc_" + args.target]
        ]
    )

    # Print overall accuracy
    print(
        f"accuracy: {(results['real_'+args.target] == results['calc_'+args.target]).mean()} ({sum(results["real_" + args.target] != results["calc_" + args.target])}/{len(results)})"
    )
