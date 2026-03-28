from hornero_event_classifier.config import CONFIG
import json
from typing import TypedDict
from pathlib import Path
import ffmpeg
import os


class VideoMetadata(TypedDict):
    video: str
    fps: str
    duration_s: float
    duration_f: int
    width: int
    height: int

def get_video_id(filepath: Path) -> str:
    return filepath.stem.rsplit("_", 1)[0]


def get_video_path(video_id: str) -> Path:
    return Path(CONFIG.video_root) / video_id.split("_", 1)[0] / (video_id + ".mp4")

def extract_metadata(path: str | Path) -> VideoMetadata:
    path = Path(path)
    print(path)
    video = path.stem
    probe_data = ffmpeg.probe(path)
    stream = probe_data["streams"][[s["codec_type"] for s in probe_data["streams"]].index("video")]
    return {
        "video": video,
        "fps": stream["avg_frame_rate"],
        "duration_s": float(stream["duration"]),
        "duration_f": int(float(stream["duration"]) * eval(stream["avg_frame_rate"])),
        "width": stream["width"],
        "height": stream["height"],
    }


def gen_metadata_file() -> dict[str, VideoMetadata]:
    nests = os.listdir(CONFIG.video_root)
    metadata = {}
    for nest in nests:
        vids = os.listdir(CONFIG.video_root / nest)
        for vid in vids:
            path = Path(CONFIG.video_root / nest / vid)
            metadata[path.stem] = extract_metadata(path)
    with open(CONFIG.video_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return metadata


loaded_video_metadata: dict[str, VideoMetadata] = {}
if CONFIG.video_root.exists():
    with open(CONFIG.video_metadata, "r", encoding="utf-8") as file:
        loaded_video_metadata = json.load(file)
elif CONFIG.auto_gen_video_metadata:
    loaded_video_metadata = gen_metadata_file()


def get_video_metadata(video_id: str) -> VideoMetadata:
    metadata = loaded_video_metadata.get(video_id, None)
    if metadata is None:
        video_path = get_video_path(video_id)
        if video_path.exists():
            metadata = extract_metadata(video_path)
            loaded_video_metadata[video_id] = metadata
        else:
            metadata = CONFIG.default_video_metadata
    return metadata
