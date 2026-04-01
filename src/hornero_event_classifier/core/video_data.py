from __future__ import annotations

import json
from typing import Any, Iterable
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import ffmpeg


@dataclass(frozen=True)
class VideoMetadata:
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
        return self.name.split("_", 1)[0]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def gen_metadata(data: Iterable[tuple[str | Path, str | Path]]) -> dict[str, VideoMetadata]:
    out: dict[str, VideoMetadata] = {}
    for yolo, video in data:
        yolo = Path(yolo)
        video = Path(video)
        if not yolo.is_file():
            raise FileNotFoundError(f"YOLO file does not exist: {yolo}")
        if not video.is_file():
            raise FileNotFoundError(f"Video file does not exist: {video}")
        probe_data = ffmpeg.probe(video)
        stream = probe_data["streams"][[s["codec_type"] for s in probe_data["streams"]].index("video")]
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, VideoMetadata):
        return value.as_dict()
    return value


def write_metadata(filepath: str | Path, data: dict[str, VideoMetadata]):
    with open(filepath, "w", encoding="utf-8") as file:
        return json.dump(data, file, default=_encoder, indent=2)


def _obj_hook(value: dict) -> dict | VideoMetadata:
    if "name" in value:
        value["yolo_path"] = Path(value["yolo_path"])
        value["video_path"] = Path(value["video_path"])
        return VideoMetadata(**value)
    else:
        return value


def read_metadata(filepath: str | Path):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file, object_hook=_obj_hook)
