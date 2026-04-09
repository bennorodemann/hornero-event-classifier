import json
import time
from os import remove
from pathlib import Path

import numpy as np
import pandas as pd
from paths import RESULTS_FILE

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
from hornero_event_classifier.classifiers import KMeanClassifier
from hornero_event_classifier.tools import event_plot


def _no_print(*_, **__) -> None:
    pass


def load_default_classifier() -> Classifier:
    return KMeanClassifier(
        (
            Metric.AVG_PLASTIC,
            Metric.AVG_Y_SCORE,
            Metric.RING_PRESENCE,
            Metric.RAD_STD,
            Metric.AVG_RING_CONF,
            Metric.PER_OWNERSHIP,
        )
    )
    dir_ = Path(__file__).parent
    with open(dir_ / "weights.json", "r", encoding="utf-8") as file:
        data = json.load(file)
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
    t0 = time.time()
    print_func = print if show_progress else _no_print
    filename = metadata.name
    print_func(f"{filename}: loading...", end="")
    s = Scene.from_metadata(metadata)
    print_func(f"\r\033[K{filename}: pre-processing...", end="")
    s.split_items(filters.make_gap_filter(max_bird_gap), ItemType.BIRD)
    if fill_bird_gaps:
        if dont_fill_at_edge:
            s.fill_gaps(filters.invert_filter(filters.frame_touch_filter), ItemType.BIRD)
            s.split_items(filters.make_gap_filter(2), ItemType.BIRD)
        else:
            s.fill_gaps(None, ItemType.BIRD)
    s.split_items((filters.make_buffer_filter(100), filters.boundary_filter), ItemType.BIRD)
    s.remove_low_conf(remove_low_conf, ItemType.BIRD)
    print_func(f"\r\033[K{filename}: classifying...", end="")
    s.classify(classifier).define_events(combine_events_within).remove_minor_items(min_event_len, ItemType.EVENT)
    print_func(f"\r\033[K{filename}: done ({time.time()-t0:.2f} s)")
    return s.get_results(), s


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from animate import event_plot_open_vid

    RESTART_CLASSIFICATION: bool = True

    start_time = time.time()
    out: list[pd.DataFrame] = []

    metadata_repo = read_metadata("data/video_metadata.json")

    file_exists = RESULTS_FILE.exists()
    if RESTART_CLASSIFICATION and file_exists:
        remove(RESULTS_FILE)
    elif file_exists:
        old_data = pd.read_csv(RESULTS_FILE)
        out.append(old_data)
        already_precessed = np.unique(old_data["video_id"])
        video_ids = [video_id for video_id in metadata_repo if video_id not in already_precessed]

    for file_metadata in metadata_repo.values():
        results, scene = classify(file_metadata, load_default_classifier())
        results.to_csv(RESULTS_FILE, index=False, header=not RESULTS_FILE.exists(), mode="a")
        out.append(results)

    results: pd.DataFrame = pd.concat(out)
    results = pd.read_csv(RESULTS_FILE)
    print(f"total time: {time.time()-start_time}s")
    fig, ax, interactor = event_plot(metadata_repo, results, ctrl_click_callback=event_plot_open_vid)
    fig.suptitle("Events")
    plt.show()
