"""
Path configuration constants.

This module defines the file paths and directories used throughout the
hornero event classifier project. All paths are configured relative to
the project structure.
"""

from __future__ import annotations
from pathlib import Path
from argparse import ArgumentParser, SUPPRESS
import json
from typing import Type

CONFIG_PATH: Path = Path(__file__).parent.parent / "hec-config.json"


class ConfigVariable:
    name: str

    def __init__(self, default: str | Path, abbreviation: str = "", base_path: bool = False, doc: str = "") -> None:
        self.default: str = str(default)
        self.abbreviation: str = abbreviation
        self.base_path: bool = base_path
        self.doc: str = doc

    def __set_name__(self, owner: Type[Config], name: str):
        self.name = name
        owner.config_vars.append(self)
        # owner._raw_paths[self.name] = self.default

    def __get__(self, obj: Config, objtype: Type[Config] | None = None):
        if obj is None:
            return self
        val: str = obj._raw_paths.get(self.name, self.default)
        # val: str = getattr(obj, self.raw_var_name, self.default)
        if not self.base_path:
            val = val.format(data_folder=obj.data_folder)
        return Path(val)

    def __set__(self, obj: Config, value: str | Path):
        # setattr(obj, self.raw_var_name, str(value))
        obj._raw_paths[self.name] = str(value)

    def get_str(self, obj: Config) -> str:
        full_path = str(self.__get__(obj, type(obj)))
        # raw_path = getattr(obj, self.raw_var_name)
        raw_path = obj._raw_paths.get(self.name, self.default)
        out_str = f"{self.name}: {raw_path}"
        if full_path != raw_path:
            out_str += f" ({full_path})"
        return out_str


class Config:
    config_vars: list[ConfigVariable] = []
    data_folder: ConfigVariable = ConfigVariable(
        default=Path(__file__).parent.parent / "data", base_path=True, doc="Base data directory"
    )
    yolo_folder: ConfigVariable = ConfigVariable(
        default="{data_folder}/YOLO", doc="Directory containing YOLO detection CSV files"
    )
    boris_file: ConfigVariable = ConfigVariable(
        default="{data_folder}/DB_BORIS.csv", doc="Ground truth BORIS annotation file"
    )
    metadata_file: ConfigVariable = ConfigVariable(
        default="{data_folder}/video_metadata.csv", doc="Video metadata repository file"
    )
    results_file: ConfigVariable = ConfigVariable(
        default="{data_folder}/hec_output.csv", doc="Output file for event classification results"
    )
    videos_root_path: ConfigVariable = ConfigVariable(
        default=Path.home() / "Videos/videos_BORIS",
        doc="Root directory for video files (organized by nest subdirectories)",
    )
    segments_cache_path: ConfigVariable = ConfigVariable(
        default="{data_folder}/segment_cache.csv", doc="Cache of raw segment metric values"
    )

    def __str__(self) -> str:
        out_str: str = "Config\n{\n"
        for var in self.config_vars:
            out_str += "    " + var.get_str(self) + ",\n"
        out_str += "}"
        return out_str

    def __init__(self, **overwrite_vars) -> None:
        self._raw_paths: dict[str, str] = {}
        for var, val in overwrite_vars.items():
            setattr(self, var, val)


def load_config_file() -> dict[str, str]:
    if not CONFIG_PATH.is_file():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def save_config_file(data: dict[str, str]):
    with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
        json.dump(data, config_file, indent=1)


def set_(**set_vars):
    config_file_data = load_config_file()
    config_file_data.update(set_vars)
    save_config_file(config_file_data)


def get(config: Config, *get_vars, get_all: bool = False) -> list[str]:
    if get_all:
        return [getattr(Config, var.name).get_str(config) for var in Config.config_vars]
    return [getattr(Config, var).get_str(config) for var in get_vars]


def reset(*reset_vars, reset_all: bool = False):
    if reset_all:
        config_file_data = {}
    else:
        config_file_data = load_config_file()
        config_file_data = {key: val for key, val in config_file_data.items() if key not in reset_vars}
    save_config_file(config_file_data)


config = Config(**load_config_file())
# config editor:

parser = ArgumentParser()
subparsers = parser.add_subparsers(dest="command")
get_parser = subparsers.add_parser("get")
set_parser = subparsers.add_parser("set")
reset_parser = subparsers.add_parser("reset")

get_parser.add_argument("--all", action="store_true", help="select all config variables")
reset_parser.add_argument("--all", action="store_true", help="select all config variables")
for variable in Config.config_vars:
    refs = ["--" + variable.name]
    if variable.abbreviation:
        refs.append("-" + variable.abbreviation)

    get_parser.add_argument(*refs, action="store_true", help=variable.doc, default=SUPPRESS)
    set_parser.add_argument(*refs, type=str, help=variable.doc, required=False, default=SUPPRESS)
    reset_parser.add_argument(*refs, action="store_true", help=variable.doc, default=SUPPRESS)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    cmd: str = args.pop("command")
    all_present: bool = args.pop("all", False)
    match cmd:
        case "set":
            set_(**args)
        case "get":
            var_vals: list[str] = get(config, *args.keys(), get_all=all_present)
            for var_val in var_vals:
                print(var_val)
        case "reset":
            reset(*args.keys(), reset_all=all_present)
        case _:
            print(config)
