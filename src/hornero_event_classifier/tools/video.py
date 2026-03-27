from hornero_event_classifier.config import VIDEO_METADATA_PATH, VIDEO_SOURCE_PATH, AUTO_GEN_VIDEO_METADATA
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
    return Path(VIDEO_SOURCE_PATH) / video_id.split("_", 1)[0] / (video_id + ".mp4")

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


def gen_metadata_file():
    nests = os.listdir(VIDEO_SOURCE_PATH)
    metadata = {}
    for nest in nests:
        vids = os.listdir(VIDEO_SOURCE_PATH / nest)
        for vid in vids:
            path = Path(VIDEO_SOURCE_PATH / nest / vid)
            metadata[path.stem] = extract_metadata(path)
    with open(VIDEO_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f)


loaded_video_metadata: dict[str, VideoMetadata] = {}
if VIDEO_METADATA_PATH.exists():
    with open(VIDEO_METADATA_PATH, "r", encoding="utf-8") as file:
        loaded_video_metadata = json.load(file)
elif AUTO_GEN_VIDEO_METADATA:
    gen_metadata_file()
    with open(VIDEO_METADATA_PATH, "r", encoding="utf-8") as file:
        loaded_video_metadata = json.load(file)


def get_video_metadata(video_id: str) -> VideoMetadata:
    metadata = loaded_video_metadata.get(video_id, None)
    if metadata is None:
        metadata = extract_metadata(get_video_path(video_id))
        loaded_video_metadata[video_id] = metadata
    return metadata
