from pathlib import Path

import numpy as np
import pandas as pd

import hornero_event_classifier.core.filters as filters
from hornero_event_classifier import tools
from hornero_event_classifier.animate.animate import Animator
from hornero_event_classifier.classifiers import Classifier, Metric
from hornero_event_classifier.core import ItemType, Scene


def _no_print(*_, **__) -> None:
    pass


def classify(
    metadata: tools.VideoMetadata,
    classifier: Classifier,
    show_progress: bool = True,
    max_bird_gap: int = 100,
    fill_bird_gaps: bool = True,
    dont_fill_at_edge: bool = True,
    remove_low_conf: float = 0.7,
    combine_events_within: int = 120,
    min_event_len: int = 100,
) -> tuple[pd.DataFrame, Scene]:
    print_func = print if show_progress else _no_print
    # file = Path(file)
    # filename: str = file.name
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
    print_func(f"\r\033[K{filename}: done")
    return s.get_results(), s


def animate(
    scene: Scene,
    scale: float = 1,
    frame: int | None = None,
    clip: tuple[int, int] | None = None,
    auto_play: bool = True,
    out_video: str | None = None,
):
    if clip and frame and not (clip[0] <= frame <= clip[1]):
        raise ValueError(f"frame ({frame}) needs to be between clip values {clip}")
    video_path = scene.video_data.video_path
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return
    scene.fill_gaps(None, ItemType.EVENT)
    with Animator(scene, out_video, scale=scale) as animator:
        if clip:
            animator.set_start(clip[0])
            animator.set_end(clip[1])
            animator.clipped = True
        if frame:
            animator.set_frame(frame)
        animator.paused = not auto_play
        animator.display_frames()


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


def validate_events(
    yolo_data: pd.DataFrame,
    boris_data: pd.DataFrame,
    overlap_threshold: float = 0.8,
    print_results: bool = True,
    long_print: bool = False,
) -> pd.DataFrame:
    overlaps: pd.DataFrame = tools.get_overlap(yolo_data, boris_data)
    grades: pd.DataFrame = tools.grade_events(overlaps, overlap=overlap_threshold)
    if print_results:
        print(tools.event_validation_str(grades, long_print))
    return grades
