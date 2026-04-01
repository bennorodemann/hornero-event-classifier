import os
from typing import Any
from pathlib import Path
from hornero_event_classifier.tools import gen_metadata, VideoMetadata, write_metadata
from paths import METADATA_FILE, YOLO_FOLDER, VIDEOS_ROOT_PATH


def video_sub_path(video_id: str) -> str:
    nest = video_id.split("_", 1)[0]
    return f"{nest}/{video_id}.mp4"


yolo_files: list[Path] = [YOLO_FOLDER / file for file in os.listdir(YOLO_FOLDER)]
video_files: list[Path] = [VIDEOS_ROOT_PATH / video_sub_path(file.stem.rsplit("_", 1)[0]) for file in yolo_files]


def serializer(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, VideoMetadata):
        return value.as_dict()
    return value


metadata = gen_metadata(zip(yolo_files, video_files))
write_metadata(METADATA_FILE, metadata)
