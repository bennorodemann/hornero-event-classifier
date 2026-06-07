"""Video metadata extraction and CSV serialization helpers.

This module relies on ``ffmpeg``/``ffprobe`` being available on the system path (via ``ffmpeg-python``).
"""

from __future__ import annotations

from csv import DictReader, DictWriter
from typing import Any, Iterable, Iterator, Self
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import ffmpeg
from collections.abc import Mapping


@dataclass(frozen=True)
class VideoMetadata(Mapping):
    """Metadata describing a single video and its corresponding YOLO file.

    :param nest: Nest id
    :type nest: str
    :param file_name: Video file name (with extension).
    :type file_name: str
    :param fps: Frames-per-second as a string fraction (e.g., ``"30/1"``).
    :type fps: str
    :param duration_f: Duration in frames.
    :type duration_f: int
    :param width: Frame width in pixels.
    :type width: int
    :param height: Frame height in pixels.
    :type height: int
    :param yolo_path: Absolute path to the YOLO CSV file.
    :type yolo_path: Path
    :param file_path: Absolute path to the video file.
    :type file_path: Path
    """

    nest_id: str
    file_name: str
    fps: str
    duration_f: int
    width: int
    height: int
    yolo_path: Path
    file_path: Path

    @classmethod
    def read_row(cls, row: dict[str, str]) -> Self:
        """Create ``VideoMetadata`` object from csv ``dict``"""
        return cls(
            nest_id=row["nest_id"],
            file_name=row["file_name"],
            fps=row["fps"],
            duration_f=int(row["duration_f"]),
            width=int(row["width"]),
            height=int(row["height"]),
            yolo_path=Path(row["yolo_path"]),
            file_path=Path(row["file_path"]),
        )

    @property
    def name(self) -> str:
        """Returns ``self.video_file`` without the file extension"""
        return self.file_name.rsplit(".", 1)[0]

    def as_dict(self) -> dict[str, Any]:
        """Return a CSV-serializable dict representation."""
        return asdict(self)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return (f.name for f in fields(self))

    def __len__(self) -> int:
        return len(fields(self))


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
        nest = video.stem.split("_", 1)[0]
        out[video.stem] = VideoMetadata(
            nest_id=nest,
            file_name=video.name,
            fps=stream["avg_frame_rate"],
            duration_f=int(float(stream["duration"]) * eval(stream["avg_frame_rate"])),
            width=stream["width"],
            height=stream["height"],
            yolo_path=yolo.absolute(),
            file_path=video.absolute(),
        )
    return out


def write_metadata(filepath: str | Path, data: dict[str, VideoMetadata]):
    """Write metadata to a CSV file.

    :param filepath: Output file path.
    :type filepath: str | Path
    :param data: Metadata mapping.
    :type data: dict[str, VideoMetadata]
    """

    with open(filepath, "w", encoding="utf-8") as file:
        writer = DictWriter(file, fieldnames=[f.name for f in fields(VideoMetadata)])
        writer.writeheader()
        writer.writerows(data.values())


def read_metadata(filepath: str | Path) -> dict[str, VideoMetadata]:
    """Read metadata from a CSV file.

    :param filepath: Input file path.
    :type filepath: str | Path
    :return: Parsed metadata mapping.
    :rtype: dict[str, VideoMetadata]
    """
    with open(filepath, "r", encoding="utf-8") as file:
        metadata = [VideoMetadata.read_row(row) for row in DictReader(file)]
        return {data.name: data for data in metadata}
