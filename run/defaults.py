"""
Path configuration constants.

This module defines the file paths and directories used throughout the
hornero event classifier project. All paths are configured relative to
the project structure.
"""

from pathlib import Path

# Base data directory
DATA_FOLDER = Path(__file__).parent.parent / "data"

# Directory containing YOLO detection CSV files
YOLO_FOLDER: Path = DATA_FOLDER / "YOLO"
if not YOLO_FOLDER.exists():
    raise FileNotFoundError(f"No yolo data could be found at: {YOLO_FOLDER}")

# Ground truth BORIS annotation file
BORIS_FILE: Path = DATA_FOLDER / "DB_BORIS.csv"

# Video metadata repository file
METADATA_FILE: Path = DATA_FOLDER / "video_metadata.json"

# Output file for event classification results
RESULTS_FILE: Path = DATA_FOLDER / "hec_output.csv"

# Root directory for video files (organized by nest subdirectories)
VIDEOS_ROOT_PATH: Path = Path.home() / "/media/alexchan/MSc Lucio/ValidationVideos/videos"

# Cache of raw segment metric values
SEGMENTS_CACHE_PATH = DATA_FOLDER / "segment_cache.csv"
