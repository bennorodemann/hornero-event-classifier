"""
Validation script for event classification results.

This script validates the performance of the event classifier by comparing
detected events against ground truth BORIS annotations. It calculates
precision, recall, F1-score, and accuracy metrics, and provides both
summary and per-video statistics.
"""

import numpy as np
import pandas as pd
from config import config
from argparse import ArgumentParser
from typing import Collection

from hornero_event_classifier import read_metadata, VideoMetadata
from hornero_event_classifier.tools import validate_events, event_validation_plot

import matplotlib.pyplot as plt
from animate import event_plot_open_vid


def _get_validation_text(title: str, tp: float, fp: float, fn: float, tn: float) -> str:
    """
    Generate formatted validation statistics text.

    Args:
        title: Title for the validation section.
        tp: True positive count.
        fp: False positive count.
        fn: False negative count.
        tn: True negative count.

    Returns:
        Formatted string with validation metrics.
    """
    # Convert to integers for display
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    tn = int(tn)

    # Calculate standard metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # Calculate F1 score (handle division by zero)
    if precision + recall != 0:
        f1 = 2 * ((precision * recall) / (precision + recall))
    else:
        f1 = float("nan")

    # Format results as readable text
    return f"{title}:\n\tTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n\tAccuracy: {accuracy}\n\tPrecision: {precision}\n\tRecall: {recall}\n\tF1: {f1}"


def event_validation_str(target: str, data: pd.DataFrame, long: bool = False) -> str:
    """
    Generate validation summary string from results dataframe.

    Args:
        target: name of target metric (e.g. 'subject', 'mud')
        data: DataFrame with validation results.
        long: If True, include per-video statistics.

    Returns:
        Formatted validation text.
    """
    # True negatives are not tracked per video, set to 0
    text = ""

    if long:
        # Generate per-video statistics
        video_accuracy = data.groupby("video_id")[["result"]].value_counts().unstack()
        video_accuracy = video_accuracy.replace(np.nan, 0)

        for video_name, accuracy_data in video_accuracy.iterrows():
            text += (
                _get_validation_text(
                    str(video_name), accuracy_data["TP"], accuracy_data["FP"], accuracy_data["FN"], accuracy_data["TN"]
                )
                + "\n\n"
            )

    # Calculate overall statistics
    tp = sum(data[target + "_result"] == "TP")
    tn = sum(data[target + "_result"] == "TN")
    fp = sum(data[target + "_result"] == "FP")
    fn = sum(data[target + "_result"] == "FN")

    # Add summary statistics
    text += _get_validation_text(target.title() + " Validation Summary", tp, fp, fn, tn)
    return text


def validate(
    target: str,
    yolo_data: pd.DataFrame,
    boris_data: pd.DataFrame,
    metadata_repo: dict[str, VideoMetadata],
    overlap_threshold: float = 0.8,
    print_results: bool = True,
    long_print: bool = False,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Validate classification results against ground truth.

    Compares YOLO classification results with BORIS annotations to calculate
    performance metrics and optionally display interactive validation plots.

    Args:
        target: name of target metric (e.g. 'subject', 'mud')
        yolo_data: DataFrame with YOLO classification results.
        boris_data: DataFrame with ground truth BORIS annotations.
        metadata_repo: Dictionary mapping video IDs to metadata.
        overlap_threshold: Minimum overlap ratio for true positives (default: 0.8).
        print_results: Whether to print validation statistics (default: True).
        long_print: Whether to include per-video statistics (default: False).
        plot: Whether to display validation plot (default: True).

    Returns:
        DataFrame with validation results.
    """
    # Perform event validation
    results: pd.DataFrame = validate_events(yolo_data, boris_data, overlap=overlap_threshold)

    # Print results if requested
    if print_results:
        print(event_validation_str(target, results, long_print))

    # Display interactive plot if requested
    if plot:
        fig, _, _ = event_validation_plot(target, metadata_repo, results, ctrl_click_callback=event_plot_open_vid)
        fig.suptitle(f"Event Validation({target.title()})")
        plt.show()

    return results


parser = ArgumentParser()
parser.add_argument("target", choices=["subject", "mud"], type=str)
parser.add_argument(
    "--overlap", default=0.8, type=float, help="Required overlap between boris and yolo event, default is 0.8"
)
parser.add_argument("--no-print", action="store_true", help="Suppress printing of validation statistics.")
parser.add_argument("--print-long", action="store_true", help="Print validation statistics for each video")
parser.add_argument("--no-plot", action="store_true", help="Suppress showing of validation plot")
parser.add_argument("--white-list", nargs="+", help="Specific video_id prefixes to include in outputs")
parser.add_argument("--black-list", nargs="+", help="Removes specified video_id prefixes from outputs")


def similar_items(source: Collection[str], refs: Collection[str]) -> list[str]:
    """Checks the strings from one `Collection` start with any of the strings from the second `Collection`.

    Args:
        source: Full strings to select from.
        refs: String prefixes to check for.

    Returns:
        Items from `source` which start with any of the strings from `refs`
    """
    out: list[str] = []
    for item in source:
        if any(item.startswith(ref) for ref in refs):
            out.append(item)
    return out


if __name__ == "__main__":
    args = parser.parse_args()
    # Load BORIS ground truth data
    boris = pd.read_csv(config.boris_file)

    # Load YOLO classification results
    yolo = pd.read_csv(config.events_file)

    # Load video metadata
    metadata = read_metadata(config.metadata_file)

    # Apply white list and black list selections
    if args.white_list:
        boris = boris[boris["video_id"].isin(similar_items(boris["video_id"].unique(), args.white_list))]
    if args.black_list:
        boris = boris[~boris["video_id"].isin(similar_items(boris["video_id"].unique(), args.black_list))]

    # Run validation
    validate(
        args.target,
        yolo,
        boris,
        metadata,
        overlap_threshold=args.overlap,
        print_results=not args.no_print,
        long_print=args.print_long,
        plot=not args.no_plot,
    )
