"""
Metadata file generation script.

This script generates video metadata files by combining YOLO detection data
with corresponding video files. It creates a metadata repository that maps
video IDs to their detection data and file paths.
"""

import os
from pathlib import Path

from config import config

from hornero_event_classifier import VideoMetadata, gen_metadata, write_metadata


def video_sub_path(video_id: str) -> str:
    """
    Generate the subdirectory path for a video based on its ID.

    Videos are organized by nest number in subdirectories.

    Args:
        video_id: The video identifier (e.g., "n10_d4_c1_1_cl2").

    Returns:
        Relative path string in format "nest/video_id.mp4".
    """
    # Extract nest identifier from video ID
    nest = video_id.split("_", 1)[0]
    return f"{nest}/{video_id}.mp4"


# Collect all YOLO detection files from the YOLO folder
yolo_files: list[Path] = [config.yolo_folder / file for file in os.listdir(config.yolo_folder)]

# Generate corresponding video file paths from YOLO filenames
video_files: list[Path] = [config.videos_root_path / video_sub_path(file.stem.rsplit("_", 1)[0]) for file in yolo_files]

# Generate metadata by pairing YOLO files with video files
metadata: dict[str, VideoMetadata] = gen_metadata(zip(yolo_files, video_files))

# Write the metadata repository to the specified file
write_metadata(config.metadata_file, metadata)
