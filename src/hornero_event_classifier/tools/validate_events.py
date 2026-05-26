"""Utilities for validating detected events against BORIS annotations.

These helpers produce labeled tables suitable for internal QA and reporting.
"""

import pandas as pd
import numpy as np


def _overlap_prep(df: pd.DataFrame, keep_mud: bool = False) -> pd.DataFrame:
    """Prepare a validation dataframe for overlap matching.

    Columns added/modified:
        - id: row index (sequential)
        - length: event length in frames
        - mud: removed (unless keep_mud is True)
    """
    # remove non hornero data
    df = df[df["subject"] != "otra_ave"].copy()
    # add row id
    df["id"] = range(len(df))
    df["length"] = df["end_frame"] - df["start_frame"] + 1
    if not keep_mud:
        del df["mud"]
    return df


def _suffix_cleaner(col_name: str) -> str:
    """Remove merge suffixes from column names."""
    return col_name.replace("_yolo", "").replace("_boris", "")


def _build_match_table(
    yolo_data: pd.DataFrame, boris_data: pd.DataFrame, overlap: float, keep_mud: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge and match YOLO and BORIS events by temporal overlap.

    :return: Tuple of (full cross-join dataframe, matched-pairs dataframe).
    """
    df = pd.merge(
        _overlap_prep(yolo_data, keep_mud=keep_mud),
        _overlap_prep(boris_data, keep_mud=keep_mud),
        on="video_id",
        suffixes=("_yolo", "_boris"),
    )
    df["frame_overlap"] = np.maximum(
        0,
        np.min(df[["end_frame_yolo", "end_frame_boris"]], axis=1)
        - np.max(df[["start_frame_yolo", "start_frame_boris"]], axis=1)
        + 1,
    )
    df["overlap_boris"] = df["frame_overlap"] / df["length_boris"]
    df["overlap_yolo"] = df["frame_overlap"] / df["length_yolo"]
    df["min_overlap"] = np.min(df[["overlap_boris", "overlap_yolo"]], axis=1)
    matched = df.query(f"subject_yolo == subject_boris and min_overlap >= {overlap}")
    return df, matched


def validate_events(yolo_data: pd.DataFrame, boris_data: pd.DataFrame, overlap: float = 0.7) -> pd.DataFrame:
    """Validate :py:class:`.Scene` results against BORIS annotations by calculating event overlap.

    Output dataframe columns:
        - video_id: the video name string
        - source: if the event came from ``"YOLO"`` or ``"BORIS"``
        - subject: ``"ring"`` or ``"no_ring"``
        - start_frame: the starting frame of the event
        - end_frame: the ending frame of the event
        - result: if the event was correctly identified (``TP``/``PAIRED``) or not (``FP``/``FN``)

    Input columns expected in both ``yolo_data`` and ``boris_data``:
        - video_id
        - subject
        - start_frame
        - end_frame
        - mud

    :param yolo_data: :py:meth:`.Scene.get_results` output.
    :type yolo_data: pd.DataFrame
    :param boris_data: ground truth boris dataframe.
    :type boris_data: pd.DataFrame
    :param overlap: the minimum required overlap between events, defaults to 0.7.
    :type overlap: float, optional
    :return: A dataframe of events and their validation result.
    :rtype: pd.DataFrame
    :seealso: :py:func:`~hornero_event_classifier.tools.recommend_weights.classify_with_boris`,
        :py:meth:`~hornero_event_classifier.core.scene.Scene.get_results`
    """
    df, matched = _build_match_table(yolo_data, boris_data, overlap, keep_mud=False)

    # sub dataframe of yolo true positives
    shared_yolo = matched[["video_id", "subject_yolo", "start_frame_yolo", "end_frame_yolo"]].rename(columns=_suffix_cleaner)
    shared_yolo.insert(1, "source", "YOLO")
    shared_yolo["result"] = "TP"

    # sub dataframe of boris events that were correctly identified
    shared_boris = matched[["video_id", "subject_boris", "start_frame_boris", "end_frame_boris"]].rename(columns=_suffix_cleaner)
    shared_boris.insert(1, "source", "BORIS")
    shared_boris["result"] = "PAIRED"

    # sub dataframe of yolo events that were not correctly identified (false positives)
    missing_yolo = df[np.isin(df["id_yolo"], np.unique(matched["id_yolo"]), invert=True)]
    missing_yolo = missing_yolo[["video_id", "subject_yolo", "start_frame_yolo", "end_frame_yolo"]].rename(
        columns=_suffix_cleaner
    )
    missing_yolo = missing_yolo.drop_duplicates()
    missing_yolo.insert(1, "source", "YOLO")
    missing_yolo["result"] = "FP"

    # sub dataframe of boris events that were not correctly identified (false negatives)
    missing_boris = df[np.isin(df["id_boris"], np.unique(matched["id_boris"]), invert=True)]
    missing_boris = missing_boris[["video_id", "subject_boris", "start_frame_boris", "end_frame_boris"]].rename(
        columns=_suffix_cleaner
    )
    missing_boris = missing_boris.drop_duplicates()
    missing_boris.insert(1, "source", "BORIS")
    missing_boris["result"] = "FN"

    # combine sub dataframes and return
    return pd.concat([shared_yolo, shared_boris, missing_yolo, missing_boris])


def mud_stats(yolo_data: pd.DataFrame, boris_data: pd.DataFrame, overlap: float = 0.7) -> dict:
    """Compute mud detection accuracy statistics for matched events.

    For each matched YOLO/BORIS event pair, compares whether YOLO's mud detection
    agreed with the BORIS ground truth annotation.

    :param yolo_data: :py:meth:`.Scene.get_results` output.
    :type yolo_data: pd.DataFrame
    :param boris_data: ground truth boris dataframe.
    :type boris_data: pd.DataFrame
    :param overlap: the minimum required overlap between events, defaults to 0.7.
    :type overlap: float, optional
    :return: Dict with keys ``matched``, ``mud_tp``, ``mud_fp``, ``mud_tn``, ``mud_fn``,
        ``accuracy``, ``precision``, ``recall``.
    :rtype: dict
    :seealso: :py:func:`validate_events`
    """
    _, matched = _build_match_table(yolo_data, boris_data, overlap, keep_mud=True)

    tp = int(( matched["mud_yolo"] &  matched["mud_boris"]).sum())
    fp = int(( matched["mud_yolo"] & ~matched["mud_boris"]).sum())
    tn = int((~matched["mud_yolo"] & ~matched["mud_boris"]).sum())
    fn = int((~matched["mud_yolo"] &  matched["mud_boris"]).sum())
    total = len(matched)

    return {
        "matched":   total,
        "mud_tp":    tp,
        "mud_fp":    fp,
        "mud_tn":    tn,
        "mud_fn":    fn,
        "accuracy":  (tp + tn) / total     if total        else float("nan"),
        "precision": tp / (tp + fp)        if (tp + fp)    else float("nan"),
        "recall":    tp / (tp + fn)        if (tp + fn)    else float("nan"),
    }
