"""
Video classification script for hornero event detection.

This module provides functionality to classify video segments using trained classifiers,
process video metadata, and generate event detection results. It supports batch
processing of multiple videos and integrates with the hornero event classifier framework.
"""

# FIXME: module doc string ^

import json
import time
from os import remove
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from defaults import RESULTS_FILE

from hornero_event_classifier import (
    Classifier,
    ItemType,
    Metric,
    Scene,
    ThresholdClassifier,
    VideoMetadata,
    filters,
    read_metadata,
)
from hornero_event_classifier.tools import event_plot


def _no_print(*_, **__) -> None:
    """Silent print function that does nothing - used to suppress output."""
    pass


def load_default_classifier() -> Classifier:
    """
    Load the default pre-trained classifier from weights.json.

    This function reads the classifier configuration including weights for each metric
    and the classification threshold from the weights.json file in the same directory.

    Returns:
        A configured ThresholdClassifier instance ready for classification.
    """
    # Get the directory containing this script
    dir_ = Path(__file__).parent

    # Load weights configuration from JSON file
    with open(dir_ / "weights.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Convert metric names to enum values and create classifier
    weights = {Metric[k]: v for k, v in data["weights"].items()}
    return ThresholdClassifier.from_dict(weights, data["threshold"])


def classify(
    metadata: VideoMetadata,
    classifier: Classifier,
    show_progress: bool = True,
    max_bird_gap: int = 100,
    fill_bird_gaps: bool = True,
    dont_fill_at_edge: bool = True,
    remove_low_conf: float = 0.7,
    combine_events_within: int = 120,
    min_event_len: int = 100,
) -> tuple[pd.DataFrame, Scene]:
    """
    Classify events in a video using the provided classifier.

    This function processes a video through several stages:
    1. Loading and pre-processing the video scene
    2. Filtering and gap-filling for bird detections
    3. Event classification and definition
    4. Post-processing to remove minor events

    Args:
        metadata: VideoMetadata object containing video information.
        classifier: Trained classifier to use for event detection.
        show_progress: Whether to display progress messages (default: True).
        max_bird_gap: Maximum gap size to split bird detections (default: 100).
        fill_bird_gaps: Whether to fill gaps in bird detections (default: True).
        dont_fill_at_edge: Whether to avoid filling gaps at video edges (default: True).
        remove_low_conf: Confidence threshold for removing low-confidence detections (default: 0.7).
        combine_events_within: Frame distance to combine nearby events (default: 120).
        min_event_len: Minimum length for events to keep (default: 100).

    Returns:
        Tuple of (results_dataframe, processed_scene).
    """
    # Start timing for performance measurement
    t0 = time.time()

    # Choose print function based on progress display setting
    print_func = print if show_progress else _no_print

    # Extract filename for progress messages
    filename = metadata.name
    print_func(f"{filename}: loading...", end="")

    # Create scene from video metadata
    s = Scene.from_metadata(metadata)

    # Pre-processing stage
    print_func(f"\r\033[K{filename}: pre-processing...", end="")

    # Split bird detections based on gap size
    s.split_items(filters.make_gap_filter(max_bird_gap), ItemType.BIRD)

    # Fill gaps in bird detections if enabled
    if fill_bird_gaps:
        if dont_fill_at_edge:
            # Fill gaps but avoid edges
            s.fill_gaps(filters.invert_filter(filters.frame_touch_filter), ItemType.BIRD)
            # Split again to clean up
            s.split_items(filters.make_gap_filter(2), ItemType.BIRD)
        else:
            # Fill all gaps
            s.fill_gaps(None, ItemType.BIRD)

    # Apply buffer and boundary filters to bird detections
    # FIXME
    s.split_items((filters.make_buffer_filter(100), filters.boundary_filter), ItemType.BIRD)

    # Remove low confidence bird detections
    s.remove_low_conf(remove_low_conf, ItemType.BIRD)

    # Classification stage
    print_func(f"\r\033[K{filename}: classifying...", end="")

    # Run classification, define events, and remove short events
    s.classify(classifier).define_events(combine_events_within).remove_minor_items(min_event_len, ItemType.EVENT)

    # Completion message with timing
    print_func(f"\r\033[K{filename}: done ({time.time()-t0:.2f} s)")

    # Return results dataframe and processed scene
    return s.get_results(), s


parser = ArgumentParser()
parser.add_argument("--restart", action="store_true", help="Re-classify videos that have already been classified.")
parser.add_argument("--no-plot", action="store_true", help="Suppresses showing of the results plot.")
parser.add_argument("--no-progress", action="store_true", help="Suppresses progress report print statements.")
parser.add_argument(
    "--max-bird-gap", default=100, type=int, help="Maximum gap size to split bird detections (default: 100)."
)
parser.add_argument("--no-fill", action="store_true", help="Does not fill in missing bounding boxes.")
parser.add_argument(
    "--fill-at-edge",
    action="store_true",
    help="Fills in missing bounding boxes even when it touches the edge of the frame.",
)
parser.add_argument(
    "--remove-low-conf",
    default=0.7,
    type=float,
    help="Confidence threshold for removing low-confidence detections (default: 0.7).",
)
parser.add_argument(
    "--combine-events-within", default=120, type=int, help="Frame distance to combine nearby events (default: 120)."
)
parser.add_argument("--min-event-len", default=100, type=int, help="Minimum length for events to keep (default: 100).")
if __name__ == "__main__":
    """
    Main execution block for batch video classification.

    This script processes all videos in the metadata repository, classifies events,
    saves results to CSV, and displays an interactive event plot. It supports
    resuming interrupted processing by checking for existing results.
    """
    import matplotlib.pyplot as plt
    from animate import event_plot_open_vid

    args = parser.parse_args()

    # Start timing the entire batch process
    start_time = time.time()

    # Load video metadata repository
    metadata_repo = read_metadata("data/video_metadata.json")

    # Check if results file exists for resuming processing
    file_exists = RESULTS_FILE.exists()
    already_processed: list[str] = []
    if args.restart and file_exists:
        # Remove existing file to restart
        remove(RESULTS_FILE)
    elif file_exists:
        # Load existing results and skip already processed videos
        old_data = pd.read_csv(RESULTS_FILE)
        already_processed = list(np.unique(old_data["video_id"]))
        # Filter out already processed videos (would need to modify metadata_repo iteration)

    # Process each video in the metadata repository
    for file_metadata in metadata_repo.values():
        # Only process if YOLO file found and not in already processed list
        if file_metadata.yolo_path.exists() and file_metadata.name not in already_processed:
            # Classify the current video
            results, scene = classify(
                file_metadata,
                load_default_classifier(),
                show_progress=not args.no_progress,
                max_bird_gap=args.max_bird_gap,
                fill_bird_gaps=not args.no_fill,
                dont_fill_at_edge=not args.fill_at_edge,
                remove_low_conf=args.remove_low_conf,
                combine_events_within=args.combine_events_within,
                min_event_len=args.min_event_len,
            )

            # Append results to CSV file (create if doesn't exist, append if it does)
            results.to_csv(RESULTS_FILE, index=False, header=not RESULTS_FILE.exists(), mode="a")

    if not args.no_progress:
        # Print total processing time
        print(f"total time: {time.time()-start_time}s")

    if not args.no_plot:
        # Read dataframe of all results
        all_results = pd.read_csv(RESULTS_FILE)

        # Create interactive event plot with click-to-open-video functionality
        fig, ax, interactor = event_plot(metadata_repo, all_results, ctrl_click_callback=event_plot_open_vid)
        fig.suptitle("Events")
        plt.show()
