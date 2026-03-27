from pathlib import Path

MAIN_PATH: Path = Path.home() / "Documents/Scripts/Python/ResearchModuleA/databases"
YOLO_PATH: Path = MAIN_PATH / "YOLOexp2"
RESULT_PATH: Path = MAIN_PATH / "general/hec_output.csv"
BORIS_PATH: Path = MAIN_PATH / "general/DB_BORIS.csv"
VIDEO_METADATA_PATH: Path = MAIN_PATH / "general/video_metadata.json"
VIDEO_SOURCE_PATH: Path = Path.home() / "Videos/videos_BORIS"
AUTO_GEN_VIDEO_METADATA: bool = True
