"""PySide-based trial viewer for classified hornero events."""

from __future__ import annotations

import json
import os
import sys
import traceback
from argparse import ArgumentParser
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import QPoint, QThread, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "hornero-event-classifier-mpl"))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hornero_event_classifier import ItemType, Metric, Scene, ThresholdClassifier, VideoMetadata, filters, read_metadata

RING_COLOR = "#2c7bb6"
NO_RING_COLOR = "#5eaec9"
PREVIEW_COLUMNS = 5
PREVIEW_ROWS = 2
PREVIEW_TILE_SIZE = (180, 132)
PREVIEW_FRAME_COUNT = 16
PREVIEW_FRAME_INTERVAL_MS = 180
MAX_PREVIEWS_PER_SUBJECT = 5
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "run" / "weights.json"
T = TypeVar("T")


@dataclass(frozen=True)
class ClipPreview:
    """Prepared preview frames for one classified event."""

    subject: str
    title: str
    frames: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class VideoAnalysis:
    """Computed plot and preview data for a single video."""

    metadata: VideoMetadata
    results: pd.DataFrame
    clips_by_subject: dict[str, tuple[ClipPreview, ...]]


def frame_to_pixmap(frame: np.ndarray, target_size) -> QPixmap:
    """Convert an RGB numpy frame into a scaled Qt pixmap."""
    height, width, _ = frame.shape
    image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image).scaled(
        target_size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _resolve_default_metadata_path() -> Path:
    config_path = REPO_ROOT / "hec-config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        data_root = Path(config.get("data_root", "")).expanduser()
        if "video_metadata" in config:
            return Path(config["video_metadata"].format(data_root=str(data_root))).expanduser()
    return REPO_ROOT / "data" / "video_metadata.json"


DEFAULT_METADATA_PATH = _resolve_default_metadata_path()


def load_default_classifier(weights_path: Path = DEFAULT_WEIGHTS_PATH) -> ThresholdClassifier:
    """Load the threshold classifier used by the existing run scripts."""
    with open(weights_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    weights = {Metric[name]: value for name, value in data["weights"].items()}
    return ThresholdClassifier.from_dict(weights, data["threshold"])


def classify_video(metadata: VideoMetadata) -> tuple[pd.DataFrame, Scene]:
    """Run the repository's standard classification steps for a single video."""
    scene = Scene.from_metadata(metadata)
    scene.split_items(filters.make_gap_filter(100), ItemType.BIRD)
    scene.fill_gaps(filters.invert_filter(filters.frame_touch_filter), ItemType.BIRD)
    scene.split_items(filters.make_gap_filter(2), ItemType.BIRD)
    scene.split_items((filters.make_buffer_filter(100), filters.boundary_filter), ItemType.BIRD)
    scene.remove_low_conf(0.7, ItemType.BIRD)
    scene.remove_low_conf(0.2, ItemType.MUD)
    scene.classify(load_default_classifier()).define_events(120).remove_minor_items(100, ItemType.EVENT)
    return scene.get_results(mud_min_frames=50, mud_max_gap=3), scene


def sample_frames(values: list[T], max_count: int) -> list[T]:
    """Sample up to ``max_count`` evenly spaced values from ``values``."""
    if len(values) <= max_count:
        return values
    sample_idx = np.linspace(0, len(values) - 1, max_count, dtype=int)
    return [values[idx] for idx in sample_idx]


def crop_bbox(frame: np.ndarray, bbox, margin_ratio: float = 0.25) -> np.ndarray:
    """Crop a frame around a bounding box with some context."""
    height, width = frame.shape[:2]
    margin_x = max(20, int((bbox.xmax - bbox.xmin) * margin_ratio))
    margin_y = max(20, int((bbox.ymax - bbox.ymin) * margin_ratio))
    x0 = max(0, int(bbox.xmin) - margin_x)
    x1 = min(width, int(bbox.xmax) + margin_x)
    y0 = max(0, int(bbox.ymin) - margin_y)
    y1 = min(height, int(bbox.ymax) + margin_y)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame
    cv2.rectangle(crop, (int(bbox.xmin) - x0, int(bbox.ymin) - y0), (int(bbox.xmax) - x0, int(bbox.ymax) - y0), (0, 0, 0), 2)
    return crop


def prepare_preview(event, capture: cv2.VideoCapture) -> ClipPreview | None:
    """Extract a short preview clip for a classified event."""
    available_frames = [box.frame for box in event.boxes.get_all()]
    if not available_frames:
        return None

    clip_frames: list[np.ndarray] = []
    for frame_number in sample_frames(available_frames, PREVIEW_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        bbox = event.boxes[frame_number]
        crop = crop_bbox(frame, bbox)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        clip_frames.append(rgb)

    if not clip_frames:
        return None

    return ClipPreview(
        subject=event.subject.value,
        title=f"{event.subject.value} {event.start}-{event.end}",
        frames=tuple(clip_frames),
    )


def prepare_previews(scene: Scene) -> dict[str, tuple[ClipPreview, ...]]:
    """Create up to five ring and five no_ring preview clips."""
    clips_by_subject: dict[str, list[ClipPreview]] = {"ring": [], "no_ring": []}
    video_path = scene.video_data.video_path
    if not video_path.exists():
        return {"ring": (), "no_ring": ()}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"ring": (), "no_ring": ()}

    try:
        events = sorted(
            scene.items.get(ItemType.EVENT),
            key=lambda item: (-item.track_len, item.start),
        )
        for event in events:
            subject = event.subject.value
            if subject not in clips_by_subject or len(clips_by_subject[subject]) >= MAX_PREVIEWS_PER_SUBJECT:
                continue
            preview = prepare_preview(event, capture)
            if preview is not None:
                clips_by_subject[subject].append(preview)
    finally:
        capture.release()

    return {
        "ring": tuple(clips_by_subject["ring"]),
        "no_ring": tuple(clips_by_subject["no_ring"]),
    }


def build_analysis(metadata: VideoMetadata) -> VideoAnalysis:
    """Classify a video and prepare plot and preview data."""
    results, scene = classify_video(metadata)
    return VideoAnalysis(
        metadata=metadata,
        results=results,
        clips_by_subject=prepare_previews(scene),
    )


class AnalysisWorker(QThread):
    """Background worker for video analysis."""

    analysis_ready = Signal(int, object)
    analysis_failed = Signal(int, str)

    def __init__(self, request_id: int, metadata: VideoMetadata) -> None:
        super().__init__()
        self.request_id = request_id
        self.metadata = metadata

    def run(self) -> None:
        try:
            analysis = build_analysis(self.metadata)
        except Exception:  # pragma: no cover - GUI error surface
            self.analysis_failed.emit(self.request_id, traceback.format_exc())
            return
        self.analysis_ready.emit(self.request_id, analysis)


class PreviewTile(QWidget):
    """One animated preview cell in the 5x2 grid."""

    def __init__(self, heading: str) -> None:
        super().__init__()
        self.clip: ClipPreview | None = None
        self.frame_index = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.title_label = QLabel(heading)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel("No example")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(*PREVIEW_TILE_SIZE)
        self.image_label.setFrameShape(QFrame.Shape.Box)
        self.image_label.setStyleSheet("background: #f5f5f5; color: #555;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, 1)

    def set_clip(self, clip: ClipPreview | None, placeholder_title: str) -> None:
        """Update the preview content for this tile."""
        self.clip = clip
        self.frame_index = 0
        self.title_label.setText(placeholder_title if clip is None else clip.title)
        if clip is None:
            self.image_label.setText("No example")
            self.image_label.setPixmap(QPixmap())
            return
        self.image_label.setText("")
        self._render_current_frame()

    def advance(self) -> None:
        """Advance to the next frame in the clip."""
        if self.clip is None or not self.clip.frames:
            return
        self.frame_index = (self.frame_index + 1) % len(self.clip.frames)
        self._render_current_frame()

    def _render_current_frame(self) -> None:
        assert self.clip is not None
        frame = self.clip.frames[self.frame_index]
        self.image_label.setPixmap(frame_to_pixmap(frame, self.image_label.size()))


class VideoFrameViewer(QWidget):
    """Large single-frame viewer for the selected video."""

    frame_changed = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.capture: cv2.VideoCapture | None = None
        self.metadata: VideoMetadata | None = None
        self.current_frame = 0
        self.playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.frame_label = QLabel("Select a video")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setMinimumSize(960, 540)
        self.frame_label.setFrameShape(QFrame.Shape.Box)
        self.frame_label.setStyleSheet("background: #111; color: #ddd;")

        self.play_button = QPushButton("Pause")
        self.play_button.clicked.connect(self.toggle_playback)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.frame_label, 1)
        layout.addWidget(self.play_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

    def clear(self, message: str = "Select a video") -> None:
        """Reset the viewer."""
        self.release_capture()
        self.frame_label.setText(message)
        self.frame_label.setPixmap(QPixmap())
        self.play_button.setText("Play")
        self.info_label.clear()

    def load_video(self, metadata: VideoMetadata) -> None:
        """Open a video for frame-by-frame display."""
        self.release_capture()
        self.metadata = metadata
        self.capture = cv2.VideoCapture(str(metadata.video_path))
        if self.capture is None or not self.capture.isOpened():
            self.clear(f"Video not found:\n{metadata.video_path}")
            return
        self.show_frame(0)

    def release_capture(self) -> None:
        """Close any open capture."""
        self.stop_playback()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.metadata = None
        self.current_frame = 0

    def show_frame(self, frame_number: int) -> None:
        """Display a specific frame from the open video."""
        if self.capture is None or self.metadata is None:
            return
        frame_number = max(0, min(int(frame_number), max(0, self.metadata.duration_f - 1)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.capture.read()
        if not ok:
            self.frame_label.setText(f"Could not read frame {frame_number}")
            self.frame_label.setPixmap(QPixmap())
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = frame_number
        self.frame_label.setText("")
        self.frame_label.setPixmap(frame_to_pixmap(rgb, self.frame_label.size()))
        self.info_label.setText(f"{self.metadata.name} | frame {frame_number}")
        self.frame_changed.emit(frame_number)

    def get_fps(self) -> float:
        """Return the current video's frame rate."""
        if self.metadata is None or not self.metadata.fps:
            return 30.0
        return float(Fraction(self.metadata.fps))

    def start_playback(self) -> None:
        """Start timed playback."""
        if self.capture is None or self.metadata is None:
            return
        fps = self.get_fps()
        interval_ms = max(1, int(1000 / max(fps, 1.0)))
        self.playing = True
        self.play_button.setText("Pause")
        self.timer.start(interval_ms)

    def stop_playback(self) -> None:
        """Stop timed playback."""
        self.playing = False
        self.play_button.setText("Play")
        self.timer.stop()

    def toggle_playback(self) -> None:
        """Toggle playback state."""
        if self.playing:
            self.stop_playback()
        else:
            self.start_playback()

    def advance_frame(self) -> None:
        """Advance playback by one frame and loop at the end."""
        if self.metadata is None:
            return
        next_frame = self.current_frame + 1
        if next_frame >= self.metadata.duration_f:
            next_frame = 0
        self.show_frame(next_frame)

    def resizeEvent(self, event) -> None:
        """Keep the current frame scaled to the widget size."""
        super().resizeEvent(event)
        if self.capture is not None and self.metadata is not None:
            self.show_frame(self.current_frame)

    def jump_seconds(self, seconds: float) -> None:
        """Jump forward or backward by a number of seconds."""
        if self.metadata is None:
            return
        offset = int(round(seconds * self.get_fps()))
        self.show_frame(self.current_frame + offset)


class TimelineCanvas(FigureCanvasQTAgg):
    """Matplotlib timeline for one video's ring and no_ring events."""

    frame_selected = Signal(int)
    event_clicked = Signal(int, str, object)

    def __init__(self) -> None:
        self.figure = Figure(figsize=(8, 3), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        self._current_frame = 0
        self._playhead = None
        self._dragging = False
        self._event_patches: list[tuple[Rectangle, int, str]] = []
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_drag)
        self.mpl_connect("button_release_event", self._on_release)
        self.draw_empty()

    def draw_empty(self, message: str = "Select a video to inspect") -> None:
        """Render the empty-state plot."""
        self.axes.clear()
        self.axes.text(0.5, 0.5, message, ha="center", va="center", transform=self.axes.transAxes, color="#555")
        self.axes.set_axis_off()
        self._playhead = None
        self._event_patches = []
        self.draw_idle()

    def plot_results(self, metadata: VideoMetadata, results: pd.DataFrame, current_frame: int = 0) -> None:
        """Plot ring and no_ring event intervals."""
        self.axes.clear()
        self.axes.set_axis_on()
        self._event_patches = []
        lanes = [("no_ring", 0, NO_RING_COLOR), ("ring", 1, RING_COLOR)]

        lane_lookup = {subject: (y, color) for subject, y, color in lanes}
        for row_index, row in results.iterrows():
            subject = row["subject"]
            y, color = lane_lookup[subject]
            start_frame = int(row["start_frame"])
            duration = int(row["end_frame"]) - start_frame + 1
            patch = Rectangle((start_frame, y - 0.35), duration, 0.7, facecolor=color, edgecolor="black")
            self.axes.add_patch(patch)
            self._event_patches.append((patch, row_index, subject))

        self.axes.set_xlim(0, max(1, metadata.duration_f))
        self.axes.set_ylim(-0.75, 1.75)
        self.axes.set_yticks([0, 1], ["no_ring", "ring"])
        self.axes.set_xlabel("Frame")
        self.axes.set_title(f"{metadata.name} event timeline")
        self.axes.grid(axis="x", alpha=0.25)
        self._playhead = self.axes.axvline(current_frame, color="#d7191c", linewidth=2)
        self._current_frame = current_frame
        self.draw_idle()

    def set_current_frame(self, frame_number: int) -> None:
        """Move the red playhead to a new frame."""
        self._current_frame = int(frame_number)
        if self._playhead is not None:
            self._playhead.set_xdata([self._current_frame, self._current_frame])
            self.draw_idle()

    def _on_click(self, event) -> None:
        """Emit the clicked frame number when the timeline is clicked."""
        if event.inaxes != self.axes or event.xdata is None:
            return
        for patch, row_index, subject in self._event_patches:
            if patch.contains(event)[0]:
                self.event_clicked.emit(row_index, subject, event.guiEvent)
                return
        self._dragging = True
        self.frame_selected.emit(max(0, int(round(event.xdata))))

    def _on_drag(self, event) -> None:
        """Scrub frames while dragging across the timeline."""
        if not self._dragging or event.inaxes != self.axes or event.xdata is None:
            return
        self.frame_selected.emit(max(0, int(round(event.xdata))))

    def _on_release(self, _event) -> None:
        """Stop dragging when the mouse button is released."""
        self._dragging = False


class TrialViewerWindow(QMainWindow):
    """Main application window."""

    def __init__(self, metadata_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Trial Viewer")
        self.resize(1600, 920)

        self.metadata_repo: dict[str, VideoMetadata] = {}
        self.analysis_cache: dict[str, VideoAnalysis] = {}
        self.current_analysis: VideoAnalysis | None = None
        self.current_frame = 0
        self.request_id = 0
        self.active_workers: set[AnalysisWorker] = set()
        self.metadata_path: Path = metadata_path or DEFAULT_METADATA_PATH

        self._build_ui()
        self.load_metadata()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)

        metadata_group = QGroupBox("Metadata")
        metadata_layout = QVBoxLayout(metadata_group)

        self.metadata_source_label = QLabel(f"Metadata: {self.metadata_path}")
        self.metadata_source_label.setWordWrap(True)
        self.load_button = QPushButton("Load metadata")
        self.browse_button = QPushButton("Browse")
        path_row = QHBoxLayout()
        path_row.addWidget(self.browse_button)
        path_row.addWidget(self.load_button)
        metadata_layout.addWidget(self.metadata_source_label)
        metadata_layout.addLayout(path_row)

        self.video_combo = QComboBox()
        self.video_combo.setEnabled(False)
        metadata_layout.addWidget(QLabel("Video"))
        metadata_layout.addWidget(self.video_combo)

        self.status_label = QLabel("Waiting for metadata")
        self.status_label.setWordWrap(True)
        metadata_layout.addWidget(self.status_label)

        self.video_info_label = QLabel("")
        self.video_info_label.setWordWrap(True)
        metadata_layout.addWidget(self.video_info_label)

        left_layout.addWidget(metadata_group)
        left_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)

        self.mode_tabs = QTabWidget()
        previews_group = QGroupBox("Event previews")
        previews_layout = QGridLayout(previews_group)
        self.preview_tiles: dict[str, list[PreviewTile]] = {"ring": [], "no_ring": []}

        for col in range(PREVIEW_COLUMNS):
            ring_tile = PreviewTile(f"ring {col + 1}")
            no_ring_tile = PreviewTile(f"no_ring {col + 1}")
            previews_layout.addWidget(ring_tile, 0, col)
            previews_layout.addWidget(no_ring_tile, 1, col)
            self.preview_tiles["ring"].append(ring_tile)
            self.preview_tiles["no_ring"].append(no_ring_tile)

        self.video_viewer = VideoFrameViewer()
        self.mode_tabs.addTab(previews_group, "Preview")
        self.mode_tabs.addTab(self.video_viewer, "Video")

        self.timeline_canvas = TimelineCanvas()

        right_layout.addWidget(self.mode_tabs, 3)
        right_layout.addWidget(self.timeline_canvas, 2)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([320, 1280])

        self.load_button.clicked.connect(self.load_metadata)
        self.browse_button.clicked.connect(self.browse_metadata)
        self.video_combo.currentIndexChanged.connect(self.on_video_selected)
        self.timeline_canvas.frame_selected.connect(self.on_timeline_frame_selected)
        self.timeline_canvas.event_clicked.connect(self.on_timeline_event_clicked)
        self.mode_tabs.currentChanged.connect(self.on_mode_changed)
        self.video_viewer.frame_changed.connect(self.on_video_frame_changed)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.advance_previews)
        self.preview_timer.start(PREVIEW_FRAME_INTERVAL_MS)

    def browse_metadata(self) -> None:
        """Open a file picker for metadata."""
        current_path = Path(self.metadata_path).expanduser()
        start_dir = current_path.parent if current_path.parent.exists() else Path.home()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select metadata JSON",
            str(start_dir),
            "JSON files (*.json)",
        )
        if path:
            self.metadata_path = Path(path)
            self.metadata_source_label.setText(f"Metadata: {self.metadata_path}")
            self.load_metadata()

    def load_metadata(self) -> None:
        """Load the metadata file into the dropdown."""
        metadata_path = Path(self.metadata_path).expanduser()
        try:
            self.metadata_repo = read_metadata(metadata_path)
        except Exception as exc:
            self.metadata_repo = {}
            self.video_combo.clear()
            self.video_combo.setEnabled(False)
            self.status_label.setText(f"Failed to load metadata: {exc}")
            self.video_info_label.clear()
            self.timeline_canvas.draw_empty("Metadata could not be loaded")
            self.video_viewer.clear("Metadata could not be loaded")
            self.clear_previews()
            return

        self.analysis_cache.clear()
        self.current_analysis = None
        self.current_frame = 0
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        self.video_combo.addItems(sorted(self.metadata_repo))
        self.video_combo.blockSignals(False)
        self.video_combo.setEnabled(bool(self.metadata_repo))
        self.status_label.setText(f"Loaded {len(self.metadata_repo)} videos from {metadata_path}")

        if self.metadata_repo:
            self.video_combo.setCurrentIndex(0)
        else:
            self.video_info_label.setText("No videos found in metadata file.")
            self.timeline_canvas.draw_empty("Metadata file is empty")
            self.video_viewer.clear("Metadata file is empty")
            self.clear_previews()

    def on_video_selected(self) -> None:
        """React to dropdown changes."""
        video_id = self.video_combo.currentText().strip()
        if not video_id or video_id not in self.metadata_repo:
            return

        metadata = self.metadata_repo[video_id]
        self.current_frame = 0
        self.video_info_label.setText(
            "\n".join(
                [
                    f"fps: {metadata.fps}",
                    f"frames: {metadata.duration_f}",
                    f"video: {metadata.video_path}",
                    f"yolo: {metadata.yolo_path}",
                ]
            )
        )

        cached = self.analysis_cache.get(video_id)
        if cached is not None:
            self.status_label.setText(
                f"Loaded cached analysis for {video_id}: {len(cached.results)} events"
            )
            self.apply_analysis(cached)
            return

        self.request_id += 1
        request_id = self.request_id
        self.status_label.setText(f"Analyzing {video_id}...")
        self.timeline_canvas.draw_empty("Analyzing selected video...")
        self.video_viewer.load_video(metadata)
        self.clear_previews(message="Loading...")

        worker = AnalysisWorker(request_id, metadata)
        worker.analysis_ready.connect(self.on_analysis_ready)
        worker.analysis_failed.connect(self.on_analysis_failed)
        worker.finished.connect(lambda: self.active_workers.discard(worker))
        self.active_workers.add(worker)
        worker.start()

    def on_analysis_ready(self, request_id: int, analysis: VideoAnalysis) -> None:
        """Handle successful background analysis."""
        if request_id != self.request_id:
            return
        self.analysis_cache[analysis.metadata.name] = analysis
        self.status_label.setText(
            f"Loaded {analysis.metadata.name}: {len(analysis.results)} events, "
            f"{len(analysis.clips_by_subject['ring'])} ring previews, "
            f"{len(analysis.clips_by_subject['no_ring'])} no_ring previews"
        )
        self.apply_analysis(analysis)

    def on_analysis_failed(self, request_id: int, details: str) -> None:
        """Handle background analysis errors."""
        if request_id != self.request_id:
            return
        self.status_label.setText("Analysis failed")
        self.timeline_canvas.draw_empty("Analysis failed")
        self.video_viewer.clear("Analysis failed")
        self.clear_previews()
        message = QMessageBox(self)
        message.setWindowTitle("Trial Viewer")
        message.setIcon(QMessageBox.Icon.Critical)
        message.setText("Failed to analyze the selected video.")
        message.setDetailedText(details)
        message.exec()

    def apply_analysis(self, analysis: VideoAnalysis) -> None:
        """Refresh plot and preview widgets from analysis data."""
        self.current_analysis = analysis
        self.video_viewer.load_video(analysis.metadata)
        self.video_viewer.show_frame(self.current_frame)
        self.timeline_canvas.plot_results(analysis.metadata, analysis.results, self.current_frame)
        self.on_mode_changed(self.mode_tabs.currentIndex())
        for subject in ("ring", "no_ring"):
            clips = analysis.clips_by_subject[subject]
            for idx, tile in enumerate(self.preview_tiles[subject]):
                clip = clips[idx] if idx < len(clips) else None
                tile.set_clip(clip, f"{subject} {idx + 1}")

    def clear_previews(self, message: str = "No example") -> None:
        """Reset the preview grid."""
        for subject in ("ring", "no_ring"):
            for idx, tile in enumerate(self.preview_tiles[subject]):
                tile.set_clip(None, f"{subject} {idx + 1}")
                tile.image_label.setText(message)

    def advance_previews(self) -> None:
        """Advance all animated preview tiles."""
        for subject in ("ring", "no_ring"):
            for tile in self.preview_tiles[subject]:
                tile.advance()

    def on_timeline_frame_selected(self, frame_number: int) -> None:
        """Jump the large video view and playhead to the clicked timeline frame."""
        self.current_frame = frame_number
        self.timeline_canvas.set_current_frame(frame_number)
        self.mode_tabs.setCurrentWidget(self.video_viewer)
        self.video_viewer.show_frame(frame_number)

    def on_timeline_event_clicked(self, row_index: int, subject: str, gui_event) -> None:
        """Show a context menu for editing a plotted event."""
        new_subject = "ring" if subject == "no_ring" else "no_ring"
        menu = QMenu(self)
        switch_action = menu.addAction(f"Switch to {new_subject}")
        if gui_event is not None and hasattr(gui_event, "globalPosition"):
            popup_pos = gui_event.globalPosition().toPoint()
        elif gui_event is not None and hasattr(gui_event, "globalPos"):
            popup_pos = gui_event.globalPos()
        else:
            popup_pos = self.timeline_canvas.mapToGlobal(QPoint(0, 0))
        chosen = menu.exec(popup_pos)
        if chosen is switch_action:
            self.apply_event_subject_change(row_index, new_subject)

    def apply_event_subject_change(self, row_index: int, new_subject: str) -> None:
        """Apply a subject change to the selected event and redraw."""
        if self.current_analysis is None:
            return
        self.current_analysis.results.at[row_index, "subject"] = new_subject
        self.timeline_canvas.plot_results(
            self.current_analysis.metadata,
            self.current_analysis.results,
            self.current_frame,
        )
        video_id = self.current_analysis.metadata.name
        ring_count = int((self.current_analysis.results["subject"] == "ring").sum())
        no_ring_count = int((self.current_analysis.results["subject"] == "no_ring").sum())
        self.status_label.setText(
            f"Edited {video_id}: {ring_count} ring events, {no_ring_count} no_ring events"
        )

    def on_mode_changed(self, index: int) -> None:
        """Start playback in video mode and stop it in preview mode."""
        if index == self.mode_tabs.indexOf(self.video_viewer):
            self.video_viewer.start_playback()
        else:
            self.video_viewer.stop_playback()

    def on_video_frame_changed(self, frame_number: int) -> None:
        """Keep the timeline playhead aligned with the video viewer."""
        self.current_frame = frame_number
        self.timeline_canvas.set_current_frame(frame_number)

    def keyPressEvent(self, event) -> None:
        """Toggle video playback with the spacebar."""
        if event.key() == Qt.Key.Key_Space:
            self.mode_tabs.setCurrentWidget(self.video_viewer)
            self.video_viewer.toggle_playback()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Left:
            self.mode_tabs.setCurrentWidget(self.video_viewer)
            self.video_viewer.jump_seconds(-5)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self.mode_tabs.setCurrentWidget(self.video_viewer)
            self.video_viewer.jump_seconds(5)
            event.accept()
            return
        super().keyPressEvent(event)


def main() -> int:
    """Launch the trial viewer."""
    parser = ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH, help="Path to a metadata JSON file.")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = TrialViewerWindow(metadata_path=args.metadata)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
