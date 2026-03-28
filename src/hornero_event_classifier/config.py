from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import __main__
import json
from typing import TYPE_CHECKING, Any, TypedDict
if TYPE_CHECKING:
    from hornero_event_classifier.tools.video import VideoMetadata

CONFIG_FILE_NAME: str = "hec-config.json"
LIBRARY_FOLDER: str = "hornero-event-classifier"

CONFIG_DEFAULTS: dict[str, Any] = {
    "data_root": "{config_folder}/data",
    "yolo_path": "{data_root}/YOLO",
    "results_path": "{data_root}/hec_output.csv",
    "boris_path": "{data_root}/DB_BORIS.csv",
    "video_metadata": "{data_root}/video_metadata.json",
    "video_root": "~/Videos/videos_BORIS",
    "auto_gen_video_metadata": True
}

class Config:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
    
    @property
    def data_root(self) -> Path:
        return self._data["data_root"]
    @property
    def yolo_path(self) -> Path:
        return self._data["yolo_path"]
    @property
    def results_path(self) -> Path:
        return self._data["results_path"]
    @property
    def boris_path(self) -> Path:
        return self._data["boris_path"]
    @property
    def video_metadata(self) -> Path:
        return self._data["video_metadata"]
    @property
    def video_root(self) -> Path:
        return self._data["video_root"]
    @property
    def auto_gen_video_metadata(self) -> bool:
        return self._data["auto_gen_video_metadata"]
    @property
    def default_video_metadata(self) -> VideoMetadata:
        return self._data["default_video_metadata"]
    
    def __getitem__(self, key):
        return self._data[key]
    

def _find_config_file():
    path: Path | str | None = getattr(__main__, "__file__", None)
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)
    if (path/CONFIG_FILE_NAME).is_file():
        out = path/CONFIG_FILE_NAME
    elif (path/".."/CONFIG_FILE_NAME).is_file():
        out = path/".."/CONFIG_FILE_NAME
    else:
        full_path: str = path.resolve().as_posix()
        cut_pos: int = full_path.find(LIBRARY_FOLDER)
        if cut_pos == -1:
            raise FileNotFoundError(f"No config file (hec-config.json) found and not in main folder ({LIBRARY_FOLDER})")
        out = Path(full_path[:cut_pos+len(LIBRARY_FOLDER)]) / CONFIG_FILE_NAME
    return out.resolve()

            
def _load_config(path: Path) -> Config:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    data["config_folder"] = path.parent
    for key, value in CONFIG_DEFAULTS.items():
        if key not in data:
            data[key] = value
    for key, value in data.items():
        if isinstance(value, str):
            value = value.format(**data)
            if value.startswith("~/"):
                data[key] = Path.home() / value[2:]
            else:
                data[key] = Path(value)
    
    return Config(data)
    
CONFIG: Config = _load_config(_find_config_file())

if __name__ == "__main__":
    print(CONFIG)