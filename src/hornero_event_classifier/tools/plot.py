from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseButton, MouseEvent
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Annotation

from hornero_event_classifier.core import VideoMetadata


class VideoBand(Rectangle):
    ALPHA = [0.1, 0.3]

    def __init__(self, metadata: VideoMetadata, y: float, **kwargs) -> None:
        self.metadata: VideoMetadata = metadata
        decor = {"fc": "k", "alpha": self.ALPHA[int(y % 2)]}
        decor.update(kwargs)
        super().__init__((0, y), self.metadata.duration_f, 1, **decor)


class EventBand(Rectangle):
    HEIGHT = 0.8 / 2
    FC: dict[str, str] = {
        "ring": "#2c7bb6",
        "no_ring": "#5eaec9",
    }
    LEGEND_DATA: list[Patch] = [
        Patch(fc=FC["ring"], label="ring"),
        Patch(fc=FC["no_ring"], label="no_ring"),
    ]

    def __init__(self, data: pd.Series, **kwargs) -> None:
        self.video_id = data["video_id"]
        self.start_frame = data["start_frame"]
        self.end_frame = data["end_frame"]
        self.length = self.end_frame - self.start_frame + 1
        y = data["video_num"] + 0.5 + self._calc_rel_y_pos(data)
        decor = self._get_decorations(data)
        decor.update(kwargs)
        super().__init__((self.start_frame, y), self.length, self.HEIGHT, **decor)

    def _calc_rel_y_pos(self, data: pd.Series) -> float:
        return -self.HEIGHT * (data["subject"] != "ring")

    def _get_decorations(self, data: pd.Series) -> dict[str, Any]:
        return {"fc": self.FC[data["subject"]], "ec": "k"}


class ValidationEventBand(EventBand):
    HEIGHT = 0.8 / 4
    FC = {
        "FP": "#d7191c",
        "FN": "#fdae61",
        "PAIRED": "#5eaec9",
        "TP": "#2c7bb6",
    }
    EC = {
        "no_ring": "w",
        "ring": "k",
    }
    LEGEND_DATA = [
        Patch(fc=FC["TP"], label="TP (YOLO)"),
        Patch(fc=FC["PAIRED"], label="TP (BORIS)"),
        Patch(fc=FC["FN"], label="FN (BORIS)"),
        Patch(fc=FC["FP"], label="FP (YOLO)"),
        Patch(fc="0.4", ec=EC["ring"], label="ringed"),
        Patch(fc="0.4", ec=EC["no_ring"], label="not ringed"),
    ]

    def _calc_rel_y_pos(self, data: pd.Series) -> float:
        is_yolo = data["source"] == "YOLO"
        is_ring = data["subject"] == "ring"
        if is_yolo and is_ring:
            return self.HEIGHT
        if is_yolo and not is_ring:
            return -2 * self.HEIGHT
        if not is_yolo and not is_ring:
            return -self.HEIGHT
        return 0

    def _get_decorations(self, data: pd.Series) -> dict[str, Any]:
        return {"fc": self.FC[data["result"]], "ec": self.EC[data["subject"]]}


class EventPlot:
    def __init__(self, metadata_repo: dict[str, VideoMetadata], df: pd.DataFrame) -> None:
        self.plot_type: str = "validation" if "source" in df.columns and "result" in df.columns else "event"

        end: int = 0
        df = df.copy()
        df["video_id"] = pd.Categorical(df["video_id"])
        df["video_num"] = df["video_id"].factorize()[0]

        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(constrained_layout=True)
        self.videos: list[VideoBand] = []
        for y, v in enumerate(df.video_id.dtype.categories):  # type: ignore
            metadata: VideoMetadata = metadata_repo[v]
            if end <= metadata.duration_f:
                end = metadata.duration_f
            video = VideoBand(metadata_repo[v], y)
            self.ax.add_patch(video)
            self.videos.append(video)

        band_type: type[EventBand] = EventBand if self.plot_type == "event" else ValidationEventBand
        self.events: dict[int, list[EventBand]] = {}
        for _, row in df.iterrows():
            event: EventBand = band_type(row)
            self.ax.add_patch(event)
            video_num: int = row["video_num"]
            if video_num not in self.events:
                self.events[video_num] = []
            self.events[video_num].append(event)

        self.ax.legend(
            title="Results",
            handles=band_type.LEGEND_DATA,
            loc="center right",
            frameon=True,
            bbox_to_anchor=(1.07, 0.6),
        )
        self.ax.set_yticks(np.arange(len(df["video_id"].dtype.categories)) + 0.5, df["video_id"].dtype.categories)  # type: ignore
        self.ax.set_xlim(0, end, auto=True)
        self.ax.set_ylim(0, max(df["video_num"]) + 1, auto=False)
        title: str = "Events" if self.plot_type == "event" else "Validation plot"
        self.fig.suptitle(title)

        self.annotation: Annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points", bbox=dict(boxstyle="round"), visible=False
        )
        self._open_func: Callable[[VideoMetadata, int], Any] = lambda _, __: _

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def set_title(self, title: str) -> None:
        self.fig.suptitle(title)

    def set_open_func(self, func: Callable[[VideoMetadata, int], Any]) -> None:
        self._open_func = func

    def show(self):
        plt.show()

    def _on_click(self, mouse_event: Event | MouseEvent):
        if not isinstance(mouse_event, MouseEvent):
            return
        if mouse_event.inaxes == self.ax and mouse_event.xdata is not None and mouse_event.ydata is not None:
            if not (0 <= int(mouse_event.ydata) < len(self.videos)):
                pass
            elif mouse_event.button == MouseButton.LEFT and mouse_event.key == "control":
                if self.videos[int(mouse_event.ydata)].contains(mouse_event)[0]:
                    self._open_func(self.videos[int(mouse_event.ydata)].metadata, int(mouse_event.xdata))
                    return
            elif mouse_event.button == MouseButton.LEFT and mouse_event.ydata is not None:
                for event in self.events[int(mouse_event.ydata)]:
                    contains, _ = event.contains(mouse_event)
                    if contains:
                        self.annotation.xy = (mouse_event.xdata, mouse_event.ydata)
                        label = f"({event.length})\n{event.start_frame} -> {event.end_frame}"
                        self.annotation.set_text(label)
                        self.annotation.set_visible(True)

                        self.annotation.set_backgroundcolor(event.get_facecolor())
                        self.fig.canvas.draw_idle()
                        return
            self.annotation.set_visible(False)
            self.fig.canvas.draw_idle()
