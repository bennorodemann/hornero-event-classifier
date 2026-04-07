import numpy as np
import pandas as pd
from paths import BORIS_FILE, METADATA_FILE, RESULTS_FILE

from hornero_event_classifier import read_metadata, VideoMetadata
from hornero_event_classifier.tools import validate_events, event_validation_plot

import matplotlib.pyplot as plt
from animate import event_plot_open_vid


def _get_validation_text(title: str, tp: float, fp: float, fn: float, tn: float) -> str:
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)
    tn = int(tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall != 0:
        f1 = 2 * ((precision * recall) / (precision + recall))
    else:
        f1 = float("nan")
    return f"{title}:\n\tTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n\tAccuracy: {accuracy}\n\tPrecision: {precision}\n\tRecall: {recall}\n\tF1: {f1}"


def event_validation_str(data: pd.DataFrame, long: bool = False) -> str:
    tn = 0
    text = ""
    if long:
        video_accuracy = data.groupby("video_id")[["result"]].value_counts().unstack()
        video_accuracy = video_accuracy.replace(np.nan, 0)
        for video_name, accuracy_data in video_accuracy.iterrows():
            text += (
                _get_validation_text(str(video_name), accuracy_data["TP"], accuracy_data["FP"], accuracy_data["FN"], tn) + "\n\n"
            )
    tp = sum(data["result"] == "TP")
    fp = sum(data["result"] == "FP")
    fn = sum(data["result"] == "FN")
    text += _get_validation_text("Summary", tp, fp, fn, tn)
    return text


def validate(
    yolo_data: pd.DataFrame,
    boris_data: pd.DataFrame,
    metadata_repo: dict[str, VideoMetadata],
    overlap_threshold: float = 0.8,
    print_results: bool = True,
    long_print: bool = False,
    plot: bool = True,
) -> pd.DataFrame:
    results: pd.DataFrame = validate_events(yolo_data, boris_data, overlap=overlap_threshold)
    if print_results:
        print(event_validation_str(results, long_print))
    if plot:
        fig, _, _ = event_validation_plot(metadata_repo, results, ctrl_click_callback=event_plot_open_vid)
        fig.suptitle("Event Validation")
        plt.show()
    return results


if __name__ == "__main__":
    boris = pd.read_csv(BORIS_FILE)
    df = pd.read_csv(RESULTS_FILE)
    metadata_repo = read_metadata(METADATA_FILE)
    results = validate(df, boris, metadata_repo)
