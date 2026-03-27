import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, Event, MouseEvent
from hornero_event_classifier.animate.animate import Animator
from hornero_event_classifier.core.data import ItemType
from hornero_event_classifier.tools.video import get_video_metadata
from typing import Literal, Any
import hornero_event_classifier as hec


class VideoBand(Rectangle):
    ALPHA = [0.1, 0.3]

    def __init__(self, video_id: str, xy: tuple[float, float], width: float, height: float, **kwargs) -> None:
        self.video_id: str = video_id
        decor = {"fc": "k", "alpha": self.ALPHA[int(xy[1] % 2)]}
        decor.update(kwargs)
        super().__init__(xy, width, height, **decor)


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


def plot_events(data: pd.DataFrame) -> tuple[Figure, Axes]:
    plot_type = "validation" if "source" in data.columns and "result" in data.columns else "event"

    end = 0
    data = data.copy()
    data["video_id"] = pd.Categorical(data["video_id"])
    data["video_num"] = data["video_id"].factorize()[0]

    fig, ax = plt.subplots(constrained_layout=True)
    videos = []
    for y, v in enumerate(data.video_id.dtype.categories):  # type: ignore
        metadata = get_video_metadata(v)
        if end <= metadata["duration_f"]:
            end = metadata["duration_f"]
        video = VideoBand(v, (0, y), metadata["duration_f"], 1)
        ax.add_patch(video)
        videos.append(video)

    band_type = EventBand if plot_type == "event" else ValidationEventBand
    events: dict[int, list[EventBand]] = {}
    for _, row in data.iterrows():
        event = band_type(row)
        ax.add_patch(event)
        video_num = row["video_num"]
        if video_num not in events:
            events[video_num] = []
        events[video_num].append(event)

    ax.legend(
        title="Results",
        handles=band_type.LEGEND_DATA,
        loc="center right",
        frameon=True,
        bbox_to_anchor=(1.07, 0.6),
    )
    ax.set_yticks(np.arange(len(data["video_id"].dtype.categories)) + 0.5, data["video_id"].dtype.categories)  # type: ignore
    ax.set_xlim(0, end, auto=True)
    ax.set_ylim(0, max(data["video_num"]) + 1, auto=False)
    title = "Events" if plot_type == "event" else "Validation plot"
    fig.suptitle(title)

    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points", bbox=dict(boxstyle="round"), visible=False)

    def on_click(mouse_event: Event | MouseEvent):
        if not isinstance(mouse_event, MouseEvent):
            return
        if mouse_event.inaxes == ax and mouse_event.xdata is not None and mouse_event.ydata is not None:
            if not (0 < int(mouse_event.ydata) < len(videos)):
                pass
            elif mouse_event.button == MouseButton.LEFT and mouse_event.key == "control":

                if videos[int(mouse_event.ydata)].contains(mouse_event)[0]:
                    hec.pipelines.animate(
                        hec.config.YOLO_PATH / f"{videos[int(mouse_event.ydata)].video_id}_bbox.csv",
                        frame=int(mouse_event.xdata),
                        auto_play=False,
                    )
                    # _, s = hec.pipelines.classify()
                    # s.fill_gaps(None, ItemType.EVENT)
                    # with Animator(s) as a:
                    #     a.paused = True
                    #     a.renderer.jump_to(int(mouse_event.xdata))
                    #     a.display_frames()
                    return
            elif mouse_event.button == MouseButton.LEFT and mouse_event.ydata is not None:
                for event in events[int(mouse_event.ydata)]:
                    contains, _ = event.contains(mouse_event)
                    if contains:
                        annot.xy = (mouse_event.xdata, mouse_event.ydata)
                        label = f"({event.length})\n{event.start_frame} -> {event.end_frame}"
                        annot.set_text(label)
                        annot.set_visible(True)

                        annot.set_backgroundcolor(event.get_facecolor())
                        fig.canvas.draw_idle()
                        return
        annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    return fig, ax
