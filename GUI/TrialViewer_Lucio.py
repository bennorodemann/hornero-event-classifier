"""Video review GUI for hornero event classifier.

Layout
------
Left  : video dropdown + metadata info
Top   : OpenCV video player with frame slider
Bottom: matplotlib detection/event timeline, click-to-seek
"""

import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hornero_event_classifier import VideoMetadata, read_metadata
from hornero_event_classifier.tools.plot import EventBand, VideoBand

METADATA_PATH = Path(__file__).parent.parent / "data" / "video_metadata.json"
RESULTS_PATH = Path(__file__).parent.parent / "data" / "hec_output.csv"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_events(video_id: str) -> pd.DataFrame | None:
    """Load classifier results for *video_id* from hec_output.csv, or None."""
    if not RESULTS_PATH.exists():
        return None
    df = pd.read_csv(RESULTS_PATH)
    rows = df[df["video_id"] == video_id]
    return rows if not rows.empty else None


# ---------------------------------------------------------------------------
# Video player widget
# ---------------------------------------------------------------------------

class VideoPlayer(QWidget):
    frame_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fps = 30.0
        self._total = 0
        self._seeking = False

        self._player = QMediaPlayer()
        self._audio = QAudioOutput()
        self._player.setAudioOutput(self._audio)
        self._player.positionChanged.connect(self._on_position)
        self._player.playbackStateChanged.connect(self._on_state)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._video = QVideoWidget()
        self._video.setMinimumSize(640, 360)
        self._video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._player.setVideoOutput(self._video)
        layout.addWidget(self._video, stretch=1)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(4, 0, 4, 0)
        rl.setSpacing(6)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(40)
        self._play_btn.clicked.connect(self.toggle_play)
        rl.addWidget(self._play_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self._slider.sliderReleased.connect(self._on_released)
        rl.addWidget(self._slider, stretch=1)

        self._lbl = QLabel("— / —")
        self._lbl.setFixedWidth(120)
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        rl.addWidget(self._lbl)

        layout.addWidget(row)

    # public -----------------------------------------------------------------

    def load(self, meta: VideoMetadata):
        self._player.stop()
        if not meta.video_path.exists():
            self._player.setSource(QUrl())
            self._slider.setMaximum(0)
            self._lbl.setText("— / —")
            return

        try:
            n, d = meta.fps.split("/")
            self._fps = float(n) / float(d)
        except (ValueError, ZeroDivisionError):
            self._fps = 30.0
        self._total = meta.duration_f
        self._slider.setMaximum(max(0, self._total - 1))
        self._player.setSource(QUrl.fromLocalFile(str(meta.video_path)))
        self._player.pause()

    def toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def seek(self, frame: int):
        self._player.setPosition(int(frame * 1000 / self._fps))

    # private ----------------------------------------------------------------

    def _on_released(self):
        self._seeking = False
        self.seek(self._slider.value())

    def _on_position(self, ms: int):
        if self._seeking:
            return
        frame = int(ms * self._fps / 1000)
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._lbl.setText(f"{frame} / {self._total}")
        self.frame_changed.emit(frame)

    @property
    def current_frame(self) -> int:
        return int(self._player.position() * self._fps / 1000)

    def _on_state(self, state):
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText("⏸" if playing else "▶")

    def closeEvent(self, event):
        self._player.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Mud event band — same geometry as EventBand, different colours and key
# ---------------------------------------------------------------------------

class MudEventBand(EventBand):
    FC = {"mud": "#a6611a", "no_mud": "#dfc27d"}
    LEGEND_DATA = [
        Patch(fc="#a6611a", label="mud"),
        Patch(fc="#dfc27d", label="no mud"),
    ]

    def _calc_rel_y_pos(self, data):
        return -self.HEIGHT * (data["subject"] != "mud")

    def _get_decorations(self, data):
        return {"fc": self.FC.get(data["subject"], "#aaa"), "ec": "k"}


# ---------------------------------------------------------------------------
# Detection / event timeline plot
# ---------------------------------------------------------------------------

class DetectionPlot(FigureCanvas):
    seek_requested = Signal(int)
    subject_switched = Signal(int)   # emits original CSV row index
    mud_switched = Signal(int)       # emits original CSV row index
    event_deleted = Signal(int)      # emits original CSV row index
    merge_requested = Signal(int, int)  # emits (first_index, second_index)
    split_requested = Signal(int)       # emits original CSV row index

    def __init__(self, parent=None):
        self._fig = Figure(tight_layout=True)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(180)
        self._axes: list = []
        self._vlines: list = []
        self._event_records: list[tuple[EventBand, pd.Series]] = []
        self._mud_records: list[tuple[MudEventBand, pd.Series]] = []
        self._duration_f: int = 0
        self._pending_merge: pd.Series | None = None
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._empty("Select a video")

    # public -----------------------------------------------------------------

    def load(self, meta: VideoMetadata, events: pd.DataFrame | None, preserve_zoom: bool = False):
        saved_xlim = self._axes[0].get_xlim() if (preserve_zoom and self._axes) else None
        self._duration_f = meta.duration_f
        self.cancel_merge()
        self._fig.clear()
        self._axes.clear()
        self._vlines.clear()
        self._event_records.clear()
        self._mud_records.clear()

        if events is None or events.empty:
            self._empty("No events — run classify.py to generate hec_output.csv")
            return

        has_mud = "mud" in events.columns
        ax_ring, ax_mud = self._fig.subplots(2, 1, sharex=True)

        ev_df = events.copy()
        ev_df["video_num"] = 0

        # --- ring / no_ring subplot ---
        ax_ring.add_patch(VideoBand(meta, 0))
        for idx, row in ev_df.iterrows():
            band = EventBand(row)
            ax_ring.add_patch(band)
            self._event_records.append((band, row))  # type: ignore[arg-type]
        ax_ring.axhline(0.5, color="k", lw=0.5, alpha=0.4)
        ax_ring.set_yticks([0.3, 0.7], ["no_ring", "ring"])
        ax_ring.set_xlim(0, meta.duration_f)
        ax_ring.set_ylim(0, 1)
        ax_ring.set_title(meta.name, fontsize=9, pad=3)
        ax_ring.tick_params(labelsize=8)
        ax_ring.legend(handles=EventBand.LEGEND_DATA, loc="upper right",
                       fontsize=8, frameon=True)
        vl = ax_ring.axvline(0, color="#e41a1c", lw=1, alpha=0.85, zorder=5)
        self._vlines.append(vl)
        self._axes.append(ax_ring)

        # --- mud / no_mud subplot ---
        ax_mud.add_patch(VideoBand(meta, 0))
        if has_mud:
            mud_df = ev_df.copy()
            mud_df["subject"] = (
                mud_df["mud"].astype(str).str.lower()
                .map({"true": "mud", "false": "no_mud"})
                .fillna("no_mud")
            )
            for _, row in mud_df.iterrows():
                band = MudEventBand(row)
                ax_mud.add_patch(band)
                self._mud_records.append((band, row))  # type: ignore[arg-type]
        ax_mud.axhline(0.5, color="k", lw=0.5, alpha=0.4)
        ax_mud.set_yticks([0.3, 0.7], ["no mud", "mud"])
        ax_mud.set_xlim(0, meta.duration_f)
        ax_mud.set_ylim(0, 1)
        ax_mud.set_xlabel("Frame", fontsize=8)
        ax_mud.tick_params(labelsize=8)
        ax_mud.legend(handles=MudEventBand.LEGEND_DATA, loc="upper right",
                      fontsize=8, frameon=True)
        vl = ax_mud.axvline(0, color="#e41a1c", lw=1, alpha=0.85, zorder=5)
        self._vlines.append(vl)
        self._axes.append(ax_mud)

        self.draw()
        if saved_xlim is not None:
            for ax in self._axes:
                ax.set_xlim(saved_xlim)
            self.draw_idle()

    def update_frame(self, frame: int):
        if not self._vlines:
            return
        for vl in self._vlines:
            vl.set_xdata([frame, frame])
        self.draw_idle()

    # private ----------------------------------------------------------------

    def _empty(self, msg: str):
        self._event_records.clear()
        self._mud_records.clear()
        self.cancel_merge()
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, color="#888", fontsize=11)
        ax.set_axis_off()
        self.draw()

    def _on_scroll(self, event):
        if not self._axes or event.xdata is None:
            return
        ax = self._axes[0]  # sharex — controlling one controls all
        xmin, xmax = ax.get_xlim()
        factor = 1.25 ** (-event.step)   # scroll up → factor < 1 → zoom in
        cx = event.xdata
        new_min = cx - (cx - xmin) * factor
        new_max = cx + (xmax - cx) * factor
        ax.set_xlim(max(0, new_min), min(new_max, self._duration_f) if self._duration_f else new_max)
        self.draw_idle()

    def _on_click(self, event):
        if event.inaxes not in self._axes or event.xdata is None:
            return

        all_records = list(self._event_records) + list(self._mud_records)

        if self._pending_merge is not None:
            for band, row in all_records:
                if band.contains(event)[0]:
                    if int(row.name) != int(self._pending_merge.name):
                        self.merge_requested.emit(int(self._pending_merge.name), int(row.name))
                    self.cancel_merge()
                    return
            self.cancel_merge()
            return

        for band, row in self._event_records:
            if band.contains(event)[0]:
                self._show_bar_menu(row, event, self.subject_switched)
                return
        for band, row in self._mud_records:
            if band.contains(event)[0]:
                self._show_bar_menu(row, event, self.mud_switched)
                return
        self.seek_requested.emit(max(0, int(event.xdata)))

    def _show_bar_menu(self, row: pd.Series, mpl_event, switch_signal: Signal):
        current = row["subject"]
        other = current[3:] if current.startswith("no_") else f"no_{current}"
        menu = QMenu(self)
        switch_action = menu.addAction(f"Switch to {other}")
        delete_action = menu.addAction("Delete event")
        merge_action = menu.addAction("Merge")
        split_action = menu.addAction("Split at current frame")
        chosen = menu.exec(QCursor.pos())
        if chosen == switch_action:
            switch_signal.emit(int(row.name))
        elif chosen == delete_action:
            self.event_deleted.emit(int(row.name))
        elif chosen == merge_action:
            self._pending_merge = row
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif chosen == split_action:
            self.split_requested.emit(int(row.name))

    def cancel_merge(self):
        self._pending_merge = None
        self.unsetCursor()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hornero Event Viewer")
        self.resize(1280, 820)
        self._meta: dict[str, VideoMetadata] = {}
        self._current_events: pd.DataFrame | None = None
        self._undo_stack: list[pd.DataFrame] = []
        self._setup_ui()
        self._load_metadata()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # --- left panel ---
        left = QWidget()
        left.setFixedWidth(220)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        ll.addWidget(QLabel("<b>Videos</b>"))

        self._combo = QComboBox()
        self._combo.setMaxVisibleItems(30)
        self._combo.currentIndexChanged.connect(self._on_video_changed)
        ll.addWidget(self._combo)

        info = QGroupBox("Info")
        il = QVBoxLayout(info)
        il.setSpacing(3)
        self._l_fps = QLabel("FPS: —")
        self._l_dur = QLabel("Duration: —")
        self._l_res = QLabel("Resolution: —")
        self._l_yolo = QLabel("YOLO: —")
        self._l_events = QLabel("Events: —")
        for lbl in (self._l_fps, self._l_dur, self._l_res, self._l_yolo, self._l_events):
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 11px;")
            il.addWidget(lbl)
        ll.addWidget(info)
        ll.addStretch()
        root.addWidget(left)

        # --- right: vertical splitter ---
        vsplit = QSplitter(Qt.Orientation.Vertical)

        self._player = VideoPlayer()
        vsplit.addWidget(self._player)

        self._plot = DetectionPlot()
        self._plot.seek_requested.connect(self._player.seek)
        vsplit.addWidget(self._plot)

        vsplit.setSizes([520, 280])
        root.addWidget(vsplit, stretch=1)

        self._player.frame_changed.connect(self._plot.update_frame)
        self._plot.subject_switched.connect(self._on_subject_switched)
        self._plot.mud_switched.connect(self._on_mud_switched)
        self._plot.event_deleted.connect(self._on_event_deleted)
        self._plot.merge_requested.connect(self._on_merge_requested)
        self._plot.split_requested.connect(self._on_split_requested)

        sc = QShortcut(QKeySequence("N"), self)
        sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc.activated.connect(self._goto_next_visit)

        sc_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        sc_undo.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_undo.activated.connect(self._undo)

        sc_esc = QShortcut(QKeySequence("Escape"), self)
        sc_esc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc_esc.activated.connect(self._plot.cancel_merge)

    def _load_metadata(self):
        if not METADATA_PATH.exists():
            self._combo.addItem("(metadata not found)")
            return
        self._meta = read_metadata(METADATA_PATH)
        self._combo.blockSignals(True)
        for name in self._meta:
            self._combo.addItem(name)
        self._combo.blockSignals(False)
        if self._combo.count() > 0:
            self._on_video_changed(0)

    def _on_video_changed(self, index: int):
        name = self._combo.itemText(index)
        if name not in self._meta:
            return
        meta = self._meta[name]

        # update info labels
        try:
            n, d = meta.fps.split("/")
            fps_val = float(n) / float(d)
            self._l_fps.setText(f"FPS: {fps_val:.3f}")
        except Exception:
            self._l_fps.setText(f"FPS: {meta.fps}")
        mins = int(meta.duration_s // 60)
        secs = int(meta.duration_s % 60)
        self._l_dur.setText(f"Duration: {mins}m {secs}s\n({meta.duration_f} frames)")
        self._l_res.setText(f"Resolution: {meta.width}×{meta.height}")
        self._l_yolo.setText("YOLO: ✓" if meta.yolo_path.exists() else "YOLO: ✗ not found")

        self._current_events = load_events(name)
        n_ev = 0 if self._current_events is None else len(self._current_events)
        self._l_events.setText(f"Events: {n_ev}" if n_ev else "Events: none")

        self._player.load(meta)
        self._plot.load(meta, self._current_events)

    def _goto_next_visit(self):
        if self._current_events is None or self._current_events.empty:
            return
        cur = self._player.current_frame
        nxt = self._current_events[self._current_events["start_frame"] > cur]
        if not nxt.empty:
            self._player.seek(int(nxt["start_frame"].min()))

    def _on_subject_switched(self, csv_index: int):
        df = pd.read_csv(RESULTS_PATH)
        df.at[csv_index, "subject"] = (
            "no_ring" if df.at[csv_index, "subject"] == "ring" else "ring"
        )
        self._apply_csv_change(df)

    def _on_mud_switched(self, csv_index: int):
        df = pd.read_csv(RESULTS_PATH)
        df.at[csv_index, "mud"] = not df.at[csv_index, "mud"]
        self._apply_csv_change(df)

    def _on_event_deleted(self, csv_index: int):
        df = pd.read_csv(RESULTS_PATH)
        self._apply_csv_change(df.drop(index=csv_index))

    def _on_split_requested(self, csv_index: int):
        frame = self._player.current_frame
        df = pd.read_csv(RESULTS_PATH)
        row = df.loc[csv_index]
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if not (start < frame < end):
            return  # current frame not inside the event — nothing to do
        new_row = row.copy()
        df.at[csv_index, "end_frame"] = frame
        new_row["start_frame"] = frame + 1
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
        self._apply_csv_change(df)

    def _on_merge_requested(self, idx1: int, idx2: int):
        df = pd.read_csv(RESULTS_PATH)
        r1, r2 = df.loc[idx1], df.loc[idx2]
        df.at[idx1, "start_frame"] = min(r1["start_frame"], r2["start_frame"])
        df.at[idx1, "end_frame"] = max(r1["end_frame"], r2["end_frame"])
        self._apply_csv_change(df.drop(index=idx2))

    def _apply_csv_change(self, new_df: pd.DataFrame):
        self._undo_stack.append(pd.read_csv(RESULTS_PATH))
        new_df.to_csv(RESULTS_PATH, index=False)
        self._reload_events()

    def _undo(self):
        if not self._undo_stack:
            return
        self._undo_stack.pop().to_csv(RESULTS_PATH, index=False)
        self._reload_events()

    def _reload_events(self):
        name = self._combo.currentText()
        self._current_events = load_events(name)
        n_ev = 0 if self._current_events is None else len(self._current_events)
        self._l_events.setText(f"Events: {n_ev}" if n_ev else "Events: none")
        self._plot.load(self._meta[name], self._current_events, preserve_zoom=True)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
