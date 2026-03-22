from __future__ import annotations

import csv
from enum import StrEnum, auto
from typing import ClassVar, Iterable, Literal, Optional

import numpy as np
from attrs import asdict, define, field

BUFFER: int = 360
OVERLAP_THRESHOLD: float = 0.7
TP: str = "TP"
TN: str = "TN"
FP: str = "FP"
FN: str = "FN"
PAIRED: str = "PAIRED"


class Subject(StrEnum):
    RING = auto()
    NO_RING = auto()
    OTRA_AVE = auto()


@define
class Event:
    source: str
    video_id: str
    start: int = field(converter=int)
    end: int = field(converter=int)
    subject: Subject = field(converter=Subject)
    mud: bool
    closes_neighbor: float = np.nan
    _result: Optional[str] = field(default=None, alias="result", init=False)
    dur_dif: Optional[float] = field(default=None, init=False)

    FIELDS: ClassVar[tuple] = ("source", "video_id", "start", "end", "subject", "mud", "result", "dur_dif", "closes_neighbor")

    @property
    def duration(self):
        return (self.end - self.start) + 1

    @property
    def result(self):
        if self._result is None:
            return FP if self.source == "yolo" else FN
        else:
            return self._result

    @result.setter
    def result(self, val):
        self._result = val

    @classmethod
    def from_yolo(cls, video_id: str, **kwargs):
        _ = kwargs.pop("id", None)
        kwargs["mud"] = kwargs["mud"] == "True"
        return cls(source="yolo", video_id=video_id, **kwargs)

    @classmethod
    def from_boris(cls, video_id: str, start_frame: str, end_frame: str, subject: str, mud: str):
        return cls(source="boris", video_id=video_id, start=start_frame, end=end_frame, subject=subject, mud=(mud == "TRUE"))

    def overlap(self, other: Event):
        overlap = max(0, (min(self.end, other.end) - max(self.start, other.start)) + 1)
        self_overlap = overlap / self.duration
        other_overlap = overlap / other.duration
        return min(self_overlap, other_overlap)
        # if other._result is not None:
        #     return np.inf
        # if self.subject != other.subject:
        #     return np.inf
        # elif (other.start < (self.start - BUFFER)) or (other.start > (self.start + BUFFER)):
        #     return np.inf
        # else:
        #     return abs(self.start - other.start) + abs(self.duration - other.duration)

    @property
    def dict(self):
        out = asdict(self, filter=lambda a, _: a.name != "_result")
        out["result"] = self.result
        return out


def read_events(filepath: str, source: Literal["yolo", "boris"]):
    match source:
        case "yolo":
            spawn_func = Event.from_yolo
        case "boris":
            spawn_func = Event.from_boris
        case _:
            raise ValueError("source must be 'yolo' or 'boris'")
    with open(filepath, "r", newline="", encoding="utf-8") as file:
        csv_data = csv.DictReader(file)
        return [spawn_func(**row) for row in csv_data if row["subject"] != Subject.OTRA_AVE.value]


def write_events(filepath: str, events: Iterable[Event]):
    with open(filepath, "w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, Event.FIELDS)
        csv_writer.writeheader()
        csv_writer.writerows(row.dict for row in events)


def print_summary(video_id: str, events: Iterable[Event]):
    tp, tn, fp, fn = 0, 0, 0, 0
    for e in events:  # (event for event in events if ((event.source != "boris") and (event.result == "TP"))):
        match e.result:
            case "TP":
                tp += 1
            case "TN":
                tn += 1
            case "FP":
                fp += 1
            case "FN":
                fn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall != 0:
        f1 = 2 * ((precision * recall) / (precision + recall))
    else:
        f1 = float("nan")
    print(video_id + ":")
    print(f"\tTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"\tAccuracy: {accuracy}")
    print(f"\tPrecision: {precision}")
    print(f"\tRecall: {recall}")
    print(f"\tF1: {f1}")


def run(
    video_id: str, boris_events: Optional[list[Event]] = None, target_dir: str = "pYOLOevents", overlap_threshold: float = 0.9
):
    yolo_events = read_events(f"databases/{target_dir}/{video_id}_events.csv", "yolo")
    if boris_events is None:
        boris_events = read_events("databases/general/DB_BORIS.csv", "boris")
    boris_events = [e for e in boris_events if e.video_id == video_id]
    for bevent in boris_events:
        # bevent.closes_neighbor = min((ye.start - bevent.start for ye in yolo_events), key=abs)
        matched_subjects = [e for e in yolo_events if e.subject == bevent.subject]
        if len(matched_subjects) == 0:
            continue
        near_event = max(matched_subjects, key=bevent.overlap)
        if bevent.overlap(near_event) > overlap_threshold:
            bevent.result = PAIRED
            near_event.result = TP
            near_event.dur_dif = near_event.duration - bevent.duration
    write_events(f"databases/{target_dir}_validation/{video_id}.csv", (*yolo_events, *boris_events))
    print_summary(video_id, [*yolo_events, *boris_events])
    return [*yolo_events, *boris_events]


def run_all(black_list: Iterable[str] = (), target_dir: str = "pYOLOevents", overlap_threshold: float = 0.9):
    boris_events = read_events("databases/general/DB_BORIS.csv", "boris")
    res = []
    for video in {e.video_id for e in boris_events if e.video_id not in black_list}:
        new_res = run(
            video,
            [e for e in boris_events if e.video_id == video],
            target_dir=target_dir,
            overlap_threshold=overlap_threshold,
        )
        if video != "n8_d2_c1_7":
            res.extend(new_res)
    print_summary("Summary", res)


if __name__ == "__main__":
    # run("n1_d1_c2_3")
    run_all(("n8_d2_c1_7",), target_dir="pYOLOv3")  # ("n8_d2_c1_7",))
