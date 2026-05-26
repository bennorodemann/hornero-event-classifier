"""
Metadata file generation script.

This script generates video metadata files by combining YOLO detection data
with corresponding video files. It creates a metadata repository that maps
video IDs to their detection data and file paths.
"""

import os
from pathlib import Path
from typing import Any

from defaults import METADATA_FILE, VIDEOS_ROOT_PATH, YOLO_FOLDER

from hornero_event_classifier import VideoMetadata, gen_metadata, write_metadata


def video_sub_path(video_id: str,VIDEOS_ROOT_PATH) -> str:
    """
    Generate the subdirectory path for a video based on its ID.

    Videos are organized by nest number in subdirectories.

    Args:
        video_id: The video identifier (e.g., "n10_d4_c1_1_cl2").

    Returns:
        Relative path string in format "nest/video_id.mp4".
    """
    # Extract nest identifier from video ID
    # nest = video_id.split("_", 1)[0]
    # nest = "nest" #temp
    # import ipdb; ipdb.set_trace()
    vid = [vid for vid in VIDEOS_ROOT_PATH.glob("**/**") if video_id in str(vid)]
        
    return vid[0]
    # return f"{nest}/{video_id}.mp4"


# Collect all YOLO detection files from the YOLO folder
yolo_files: list[Path] = [YOLO_FOLDER / file for file in os.listdir(YOLO_FOLDER)]

# Generate corresponding video file paths from YOLO filenames

video_files: list[Path] = [VIDEOS_ROOT_PATH / video_sub_path(file.stem.rsplit("_", 1)[0],VIDEOS_ROOT_PATH) for file in yolo_files]

# import ipdb; ipdb.set_trace()


def serializer(value: Any) -> Any:
    """
    Custom JSON serializer for metadata objects.

    Handles Path and VideoMetadata objects for JSON serialization.

    Args:
        value: The value to serialize.

    Returns:
        JSON-serializable representation of the value.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, VideoMetadata):
        return value.as_dict()
    return value


# Generate metadata by pairing YOLO files with video files
metadata = gen_metadata(zip(yolo_files, video_files))

# Write the metadata repository to the specified file
write_metadata(METADATA_FILE, metadata)
