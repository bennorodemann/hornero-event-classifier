"""Video metadata extraction and JSON serialization helpers.

This module relies on ``ffmpeg``/``ffprobe`` being available on the system path (via ``ffmpeg-python``).
"""

from __future__ import annotations

import json
from typing import Any, Iterable
from dataclasses import dataclass, asdict
from pathlib import Path
import ffmpeg


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata describing a single video and its corresponding YOLO file.

    :param name: Video stem name (filename without extension).
    :type name: str
    :param fps: Frames-per-second as a string fraction (e.g., ``"30/1"``).
    :type fps: str
    :param duration_s: Duration in seconds.
    :type duration_s: float
    :param duration_f: Duration in frames.
    :type duration_f: int
    :param width: Frame width in pixels.
    :type width: int
    :param height: Frame height in pixels.
    :type height: int
    :param yolo_path: Absolute path to the YOLO CSV file.
    :type yolo_path: Path
    :param video_path: Absolute path to the video file.
    :type video_path: Path
    """

    name: str
    fps: str
    duration_s: float
    duration_f: int
    width: int
    height: int
    yolo_path: Path
    video_path: Path

    @property
    def nest(self) -> str:
        """Return the nest ID parsed from the video name (prefix before first underscore)."""
        return self.name.split("_", 1)[0]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return asdict(self)


def gen_metadata(data: Iterable[tuple[str | Path, str | Path]]) -> dict[str, VideoMetadata]:
    """Generate metadata for a list of YOLO/video file pairs.

    :param data: Iterable of ``(yolo_path, video_path)`` pairs.
    :type data: Iterable[tuple[str | Path, str | Path]]
    :return: Mapping of video stem to :py:class:`VideoMetadata`.
    :rtype: dict[str, VideoMetadata]
    :raises FileNotFoundError: If a YOLO or video file path does not exist.
    :raises ffmpeg.Error: If ``ffprobe`` fails to read video metadata.
    """
    out: dict[str, VideoMetadata] = {}
    for yolo, video in data:
        yolo = Path(yolo)
        video = Path(video)
        # ensure both paths exist
        if not yolo.is_file():
            raise FileNotFoundError(f"YOLO file does not exist: {yolo}")
        if not video.is_file():
            raise FileNotFoundError(f"Video file does not exist: {video}")
        # retrieve metadata with ffprobe (from ffmpeg)
        probe_data = ffmpeg.probe(video)
        stream = probe_data["streams"][[s["codec_type"] for s in probe_data["streams"]].index("video")]
        # create instance
        out[video.stem] = VideoMetadata(
            name=video.stem,
            fps=stream["avg_frame_rate"],
            duration_s=float(stream["duration"]),
            duration_f=int(float(stream["duration"]) * eval(stream["avg_frame_rate"])),
            width=stream["width"],
            height=stream["height"],
            yolo_path=yolo.absolute(),
            video_path=video.absolute(),
        )
    return out


def _encoder(value: Any) -> Any:
    """JSON encoder for :py:class:`Path` and :py:class:`VideoMetadata` values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, VideoMetadata):
        return value.as_dict()
    return value


def write_metadata(filepath: str | Path, data: dict[str, VideoMetadata]):
    """Write metadata to a JSON file.

    :param filepath: Output file path.
    :type filepath: str | Path
    :param data: Metadata mapping.
    :type data: dict[str, VideoMetadata]
    """
    with open(filepath, "w", encoding="utf-8") as file:
        return json.dump(data, file, default=_encoder, indent=2)


def _obj_hook(value: dict) -> dict | VideoMetadata:
    """JSON object hook to rebuild :py:class:`VideoMetadata` objects."""
    if "name" in value:
        value["yolo_path"] = Path(value["yolo_path"])
        value["video_path"] = Path(value["video_path"])
        return VideoMetadata(**value)
    else:
        return value


def read_metadata(filepath: str | Path) -> dict[str, VideoMetadata]:
    """Read metadata from a JSON file.

    :param filepath: Input file path.
    :type filepath: str | Path
    :return: Parsed metadata mapping.
    :rtype: dict[str, VideoMetadata]
    """
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file, object_hook=_obj_hook)
