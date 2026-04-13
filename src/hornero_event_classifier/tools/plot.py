"""Plotting utilities for event timelines and validation results.

This module focuses on visual inspection tools for internal workflows.
"""

from __future__ import annotations
from typing import Any, Callable, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseButton, MouseEvent
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Annotation
import cv2

from hornero_event_classifier.core import VideoMetadata


class VideoBand(Rectangle):
    """A ``matplotlib.patches.Rectangle`` subclass that holds video metadata for interactivity purposes.

    The length of the rectangle is equal to :py:attr:`.VideoMetadata.duration_f` and its height is always 1.

    :param metadata: video metadata
    :type metadata: VideoMetadata
    :param y: y position of ``VideoBand``
    :type y: float
    :param kwargs: ``matplotlib.patches.Rectangle`` aesthetic passthrough arguments
    :type kwargs: Any
    """

    ALPHA = [0.1, 0.3]

    def __init__(self, metadata: VideoMetadata, y: float, **kwargs) -> None:
        # save metadata
        self.metadata: VideoMetadata = metadata
        # default decor
        decor = {
            "fc": "k",
            "alpha": self.ALPHA[int(y % 2)],
        }
        # overwrite/add to decor values
        decor.update(kwargs)
        super().__init__((0, y), self.metadata.duration_f, 1, **decor)


class EventBand(Rectangle):
    """A ``matplotlib.patches.Rectangle`` subclass that holds event data and applies aesthetics based on event parameters.

    This class works with the data from :py:meth:`.Scene.get_results`.

    :param data: a row from a ``pandas.DataFrame`` containing event data
    :type data: pd.Series
    :param kwargs: ``matplotlib.patches.Rectangle`` aesthetic passthrough arguments
    :type kwargs: Any
    """

    #: height of ``Rectangle`` in plot
    HEIGHT = 0.8 / 2
    #: ``Rectangle`` fill color lookup dict
    FC: dict[str, str] = {
        "ring": "#2c7bb6",
        "no_ring": "#5eaec9",
    }
    #: Aesthetic naming info for legends
    LEGEND_DATA: list[Patch] = [
        Patch(fc=FC["ring"], label="ring"),
        Patch(fc=FC["no_ring"], label="no_ring"),
    ]

    def __init__(self, data: pd.Series, **kwargs) -> None:
        # save relevant info
        self.video_id = data["video_id"]
        self.start_frame = data["start_frame"]
        self.end_frame = data["end_frame"]
        self.length = self.end_frame - self.start_frame + 1
        y = data["video_num"] + 0.5 + self._calc_rel_y_pos(data)
        # get default decor
        decor = self._get_decorations(data)
        # overwrite/add decor
        decor.update(kwargs)
        super().__init__((self.start_frame, y), self.length, self.HEIGHT, **decor)

    def _calc_rel_y_pos(self, data: pd.Series) -> float:
        # calculate relative position based on data contents
        return -self.HEIGHT * (data["subject"] != "ring")

    def _get_decorations(self, data: pd.Series) -> dict[str, Any]:
        return {"fc": self.FC[data["subject"]], "ec": "k"}


class ValidationEventBand(EventBand):
    """A ``EventBand`` subclass with aesthetics specific for event validation plots.

    This class works with the data from :py:func:`.validate_events.validate_events`.

    :param data: a row from a ``pandas.DataFrame`` containing validated event data
    :type data: pd.Series
    :param kwargs: ``matplotlib.patches.Rectangle`` aesthetic passthrough arguments
    :type kwargs: Any
    """

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
        # calculate relative position based on data contents
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


class EventInteractor:
    """Holds state and callbacks for interactivity in event plots.

    Used by :py:func:`event_plot` and :py:func:`event_validation_plot`.

    :param ax: plot axes
    :type ax: Axes
    :param video_bands: :py:class:`VideoBand`\\s in ``ax`` with their index being equal to their y position
    :type video_bands: list[VideoBand]
    :param event_bands: a mapping of :py:class:`VideoBand` y position to associated :py:class:`EventBand`\\s
    :type event_bands: dict[int, list[EventBand]]
    :param ctrl_click_callback: optional function that is called when user control clicks within a :py:class:`VideoBand`, default
        is ``None``
    :type ctrl_click_callback: Optional[CtrlClickCallback]
    """

    def __init__(
        self,
        ax: Axes,
        video_bands: list[VideoBand],
        event_bands: dict[int, list[EventBand]],
        ctrl_click_callback: Optional[CtrlClickCallback] = None,
    ) -> None:
        # save data
        self.ax: Axes = ax
        self.video_bands: list[VideoBand] = video_bands
        self.event_bands: dict[int, list[EventBand]] = event_bands
        self.ctrl_click_callback: Optional[CtrlClickCallback] = ctrl_click_callback
        # initiate annotation object
        self.annotation: Annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points", bbox={"boxstyle": "round"}, visible=False
        )

        self.button_cid: int
        self.close_cid: int
        self._connect_events()

    def _connect_events(self):
        self.button_cid = self.ax.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.close_cid = self.ax.figure.canvas.mpl_connect("close_event", self._on_close)

    def _on_close(self, close_event: Event):
        cv2.destroyAllWindows()

    def _on_click(self, mouse_event: Event | MouseEvent):
        # Ensure the event is a mouse click and occurred within the plot axes.
        if not isinstance(mouse_event, MouseEvent):
            return
        if mouse_event.inaxes == self.ax and mouse_event.xdata is not None and mouse_event.ydata is not None:
            # save mouse position
            mouse_pos: tuple[float, float] = (mouse_event.xdata, mouse_event.ydata)
            # If y position is outside the visible video band range, ignore the click.
            if not 0 <= int(mouse_pos[1]) < len(self.video_bands):
                return
            # if left mouse click while holding control:
            elif mouse_event.button == MouseButton.LEFT and mouse_event.key == "control":
                # check if mouse click was on a VideoBand and that there is a ctrl_click_callback
                if self.video_bands[int(mouse_pos[1])].contains(mouse_event)[0] and self.ctrl_click_callback:
                    self.ctrl_click_callback(self, self.video_bands[int(mouse_event.ydata)].metadata, mouse_pos)
                    return
            # just left click
            elif mouse_event.button == MouseButton.LEFT:
                # check each event band within current video band
                for event in self.event_bands[int(mouse_pos[1])]:
                    # if click was within event band call event_click meth and return
                    if event.contains(mouse_event)[0]:
                        self.event_click(event, mouse_pos)
                        return
            self.miss_click(mouse_pos)

    def event_click(self, event: EventBand, mouse_pos: tuple[float, float]):
        """The method that is called when a user left clicks an :py:class:`EventBand`.

        :param event: event that was clicked on
        :type event: EventBand
        :param mouse_pos: The (x, y) coordinates of the mouse click
        :type mouse_pos: tuple[float, float]
        """
        self.annotation.xy = mouse_pos
        self.annotation.set_text(f"({event.length})\n{event.start_frame} -> {event.end_frame}")
        self.annotation.set_visible(True)

        self.annotation.set_backgroundcolor(event.get_facecolor())
        self.ax.figure.canvas.draw_idle()

    def miss_click(self, mouse_pos: tuple[float, float]):
        """Handle clicks on empty plot regions by hiding any active annotation.

        The click coordinates are intentionally ignored by this handler.

        :param mouse_pos: The (x, y) coordinates of the mouse click, ignored.
        :type mouse_pos: tuple[float, float]
        """
        del mouse_pos
        self.annotation.set_visible(False)
        self.ax.figure.canvas.draw_idle()


type CtrlClickCallback = Callable[[EventInteractor, VideoMetadata, tuple[float, float]], Any]


def _make_event_plot(
    band_type: Type[EventBand],
    metadata_repo: dict[str, VideoMetadata],
    df: pd.DataFrame,
    ctrl_click_callback: Optional[CtrlClickCallback] = None,
) -> tuple[Figure, Axes, EventInteractor]:
    """Shared implementation for event timeline plots.

    :param band_type: Event band class to render.
    :type band_type: Type[EventBand]
    :param metadata_repo: Video metadata indexed by ``video_id``.
    :type metadata_repo: dict[str, VideoMetadata]
    :param df: Event dataframe with required columns.
    :type df: pd.DataFrame
    :param ctrl_click_callback: Optional callback for ctrl-click events.
    :type ctrl_click_callback: Optional[CtrlClickCallback]
    :return: ``(figure, axes, interactor)`` tuple.
    :rtype: tuple[Figure, Axes, EventInteractor]
    """
    df = df.copy()
    df["video_id"] = pd.Categorical(df["video_id"], ordered=True)
    # get video y positions
    df["video_num"], video_categories = df["video_id"].factorize()

    fig: Figure
    ax: Axes
    # initiate plot
    fig, ax = plt.subplots(constrained_layout=True)
    videos: list[VideoBand] = []
    end: int = 0
    # for every video create video band
    for y, v in enumerate(video_categories):  # enumerate(df.video_id.dtype.categories):  # type: ignore
        metadata: VideoMetadata = metadata_repo[v]
        end = max(end, metadata.duration_f)
        video = VideoBand(metadata_repo[v], y)
        ax.add_patch(video)
        videos.append(video)
        # df.loc[df["video_id"] == metadata.name, "video_num"] = y

    events: dict[int, list[EventBand]] = {}
    # for every row create a event band
    for _, row in df.iterrows():
        event: EventBand = band_type(row)
        ax.add_patch(event)
        video_num: int = row["video_num"]
        if video_num not in events:
            events[video_num] = []
        events[video_num].append(event)

    # create legend
    ax.legend(
        title="Results",
        handles=band_type.LEGEND_DATA,
        loc="center right",
        frameon=True,
        bbox_to_anchor=(1.07, 0.6),
    )
    # add video labels
    ax.set_yticks(np.arange(len(video_categories)) + 0.5, video_categories)  # type: ignore
    # set x and y limits
    ax.set_xlim(0, end, auto=True)
    ax.set_ylim(0, max(df["video_num"]) + 1, auto=False)

    # initiate interaction coordinator
    interactor = EventInteractor(ax, video_bands=videos, event_bands=events, ctrl_click_callback=ctrl_click_callback)

    return fig, ax, interactor


def event_plot(
    metadata_repo: dict[str, VideoMetadata],
    df: pd.DataFrame,
    ctrl_click_callback: Optional[CtrlClickCallback] = None,
) -> tuple[Figure, Axes, EventInteractor]:
    """Create a plot displaying event timelines across multiple videos.

    Required ``df`` columns:
        - video_id
        - subject
        - start_frame
        - end_frame

    :param metadata_repo: A dict of video metadata within ``df``
    :type metadata_repo: dict[str, VideoMetadata]
    :param df: A ``pandas.DataFrame`` describing event data. Can be taken from :py:meth:`.Scene.get_results`
    :type df: pd.DataFrame
    :param ctrl_click_callback: A callback function to be called when a user clicks within a :py:class:`VideoBand`, defaults to
        ``None``
    :type ctrl_click_callback: Optional[CtrlClickCallback], optional
    :return: Matplotlib figure and axes objects and interactivity handler.
    :rtype: tuple[Figure, Axes, EventInteractor]

    :seealso: :py:meth:`.Scene.get_results`, :py:class:`VideoBand`
    """
    return _make_event_plot(
        band_type=EventBand, metadata_repo=metadata_repo, df=df, ctrl_click_callback=ctrl_click_callback
    )


def event_validation_plot(
    metadata_repo: dict[str, VideoMetadata],
    df: pd.DataFrame,
    ctrl_click_callback: Optional[CtrlClickCallback] = None,
) -> tuple[Figure, Axes, EventInteractor]:
    """Create a plot displaying event timelines across multiple videos alongside BORIS data with color coding for accuracy.

    Required ``df`` columns:
        - video_id
        - source
        - subject
        - start_frame
        - end_frame
        - result

    :param metadata_repo: A dict of video metadata within ``df``
    :type metadata_repo: dict[str, VideoMetadata]
    :param df: A ``pandas.DataFrame`` describing event data. Can be taken from :py:func:`.validate_events.validate_events`
    :type df: pd.DataFrame
    :param ctrl_click_callback: A callback function to be called when a user clicks within a :py:class:`VideoBand`, defaults to
        ``None``
    :type ctrl_click_callback: Optional[CtrlClickCallback], optional
    :return: Matplotlib figure and axes objects and interactivity handler.
    :rtype: tuple[Figure, Axes, EventInteractor]

    :seealso: :py:func:`~.validate_events.validate_events`, :py:class:`VideoBand`
    """
    return _make_event_plot(
        band_type=ValidationEventBand, metadata_repo=metadata_repo, df=df, ctrl_click_callback=ctrl_click_callback
    )
