import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def _overlap_prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["subject"] != "otra_ave"]
    df["id"] = range(len(df))
    df["length"] = df["end_frame"] - df["start_frame"] + 1
    del df["mud"]
    return df


def get_overlap(yolo_data: pd.DataFrame, boris_data: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(_overlap_prep(yolo_data), _overlap_prep(boris_data), on="video_id", suffixes=("_yolo", "_boris"))
    df["frame_overlap"] = np.maximum(
        0,
        np.min(df[["end_frame_yolo", "end_frame_boris"]], axis=1)
        - np.max(df[["start_frame_yolo", "start_frame_boris"]], axis=1)
        + 1,
    )
    df["overlap_boris"] = df["frame_overlap"] / df["length_boris"]
    df["overlap_yolo"] = df["frame_overlap"] / df["length_yolo"]
    return df


# def get_overhang(data: pd.DataFrame) -> pd.DataFrame:
#     yolo_overhang = data.groupby(["video_id", "id_yolo"])


def _suffix_cleaner(col_name: str) -> str:
    return col_name.replace("_yolo", "").replace("_boris", "")


def validate_events(data: pd.DataFrame, overlap: float = 0.7) -> pd.DataFrame:
    data = data.copy()
    data["min_overlap"] = np.min(data[["overlap_boris", "overlap_yolo"]], axis=1)
    shared = data.query(f"subject_yolo == subject_boris and min_overlap >= {overlap}")

    shared_yolo = shared[["video_id", "subject_yolo", "start_frame_yolo", "end_frame_yolo"]].rename(columns=_suffix_cleaner)
    shared_yolo.insert(1, "source", "YOLO")
    shared_yolo["result"] = "TP"

    shared_boris = shared[["video_id", "subject_boris", "start_frame_boris", "end_frame_boris"]].rename(columns=_suffix_cleaner)
    shared_boris.insert(1, "source", "BORIS")
    shared_boris["result"] = "PAIRED"

    missing_yolo = data[np.isin(data["id_yolo"], np.unique(shared["id_yolo"]), invert=True)]
    missing_yolo = missing_yolo[["video_id", "subject_yolo", "start_frame_yolo", "end_frame_yolo"]].rename(
        columns=_suffix_cleaner
    )
    missing_yolo = missing_yolo.drop_duplicates()
    missing_yolo.insert(1, "source", "YOLO")
    missing_yolo["result"] = "FP"

    missing_boris = data[np.isin(data["id_boris"], np.unique(shared["id_boris"]), invert=True)]
    missing_boris = missing_boris[["video_id", "subject_boris", "start_frame_boris", "end_frame_boris"]].rename(
        columns=_suffix_cleaner
    )
    missing_boris = missing_boris.drop_duplicates()
    missing_boris.insert(1, "source", "BORIS")
    missing_boris["result"] = "FN"
    return pd.concat([shared_yolo, shared_boris, missing_yolo, missing_boris])


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
    TN = 0
    text = ""
    if long:
        video_accuracy = data.groupby("video_id")[["result"]].value_counts().unstack()
        video_accuracy = video_accuracy.replace(np.nan, 0)
        for video_name, accuracy_data in video_accuracy.iterrows():
            text += (
                _get_validation_text(str(video_name), accuracy_data["TP"], accuracy_data["FP"], accuracy_data["FN"], TN) + "\n\n"
            )
    TP = sum(data["result"] == "TP")
    FP = sum(data["result"] == "FP")
    FN = sum(data["result"] == "FN")
    text += _get_validation_text("Summary", TP, FP, FN, TN)
    return text


def plot_validations(data: pd.DataFrame) -> tuple[Figure, Axes]:
    h = 0.8 / 4
    edge_color = {
        "no_ring": "w",
        "ring": "k",
    }
    face_color = {
        "FP": "#d7191c",
        "FN": "#fdae61",
        "PAIRED": "#5eaec9",
        "TP": "#2c7bb6",
    }
    end = max(data["end_frame"])
    data = data.copy()
    video_num, video_names = pd.factorize(data["video_id"])
    data["video_num"] = video_num
    has_ring = data["subject"] == "ring"
    from_yolo = data["source"] == "YOLO"
    data["y_pos"] = 0.5 + np.array([-h, 0, -2 * h, h])[has_ring + (2 * from_yolo)]

    fig, ax = plt.subplots(constrained_layout=True)
    for v in range(0, max(data["video_num"]) + 1):
        rect = Rectangle((-100, v), end + 700, 1, fc="k", alpha=0.1 + (0.2 * (not v % 2)))
        ax.add_patch(rect)
    for _, row in data.iterrows():
        rect = Rectangle(
            (row["start_frame"], row["video_num"] + row["y_pos"]),
            row["end_frame"] - row["start_frame"] + 1,
            h,
            fc=face_color[row["result"]],
            ec=edge_color[row["subject"]],
            label=row["result"],
        )
        ax.add_patch(rect)
    ax.legend(
        title="Results",
        handles=[
            Patch(fc=face_color["TP"], label="TP (YOLO)"),
            Patch(fc=face_color["PAIRED"], label="TP (BORIS)"),
            Patch(fc=face_color["FN"], label="FN (BORIS)"),
            Patch(fc=face_color["FP"], label="FP (YOLO)"),
            Patch(fc="0.4", ec=edge_color["ring"], label="ringed"),
            Patch(fc="0.4", ec=edge_color["no_ring"], label="not ringed"),
        ],
        loc="center right",
        frameon=True,
        bbox_to_anchor=(1, 0.5),
        draggable=True,
    )

    ax.set_yticks(np.arange(len(video_names)) + 0.5, video_names)
    ax.set_xlim(0, end + 500, auto=False)
    ax.set_ylim(0, max(data["video_num"]) + 1, auto=False)

    return fig, ax
