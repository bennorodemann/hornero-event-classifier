import pandas as pd
import numpy as np


def _overlap_prep(df: pd.DataFrame) -> pd.DataFrame:
    # remove non hornero data
    df = df[df["subject"] != "otra_ave"]
    # add row id
    df["id"] = range(len(df))
    df["length"] = df["end_frame"] - df["start_frame"] + 1
    del df["mud"]
    return df


def _suffix_cleaner(col_name: str) -> str:
    # remove df suffix
    return col_name.replace("_yolo", "").replace("_boris", "")


def validate_events(yolo_data: pd.DataFrame, boris_data: pd.DataFrame, overlap: float = 0.7) -> pd.DataFrame:
    """Validate :py:class:`.Scene` results using a boris validation by calculating percent overlap of between events.

    Columns:
        - video_id: the video name string
        - source: if the event came from ``"YOLO"`` or ``"BORIS"``
        - subject: ``"ring"`` or ``"no_ring"``
        - start_frame: the staring frame of event
        - end_frame: the ending frame of event
        - result: if the event was correctly identified (``TP``/``PAIRED``) or not (``FP``/``FN``)

    :param yolo_data: :py:meth:`.Scene.get_results` output.
    :type yolo_data: pd.DataFrame
    :param boris_data: ground truth boris dataframe.
    :type boris_data: pd.DataFrame
    :param overlap: the minimum required overlap between events, defaults to 0.7.
    :type overlap: float, optional
    :return: a dataframe of events and if they were correctly identified or not.
    :rtype: pd.DataFrame
    """
    # inner join by video_id
    df = pd.merge(_overlap_prep(yolo_data), _overlap_prep(boris_data), on="video_id", suffixes=("_yolo", "_boris"))
    # get number of frames that overlap
    df["frame_overlap"] = np.maximum(
        0,
        np.min(df[["end_frame_yolo", "end_frame_boris"]], axis=1)
        - np.max(df[["start_frame_yolo", "start_frame_boris"]], axis=1)
        + 1,
    )
    # percent overlap from boris and yolo perspective
    df["overlap_boris"] = df["frame_overlap"] / df["length_boris"]
    df["overlap_yolo"] = df["frame_overlap"] / df["length_yolo"]
    # get minimum overlap between both perspectives
    df["min_overlap"] = np.min(df[["overlap_boris", "overlap_yolo"]], axis=1)
    # find rows that share the same subject and have a minimum overlap over or equal to the overlap threshold
    matched = df.query(f"subject_yolo == subject_boris and min_overlap >= {overlap}")

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
