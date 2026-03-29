"""The config module automatically finds and loads a local config file. The :code:`CONFIG` variable can be imported by other
modules for a shared workspace. While there are default values for most variables, a config file with the name
:code:`hec-config.json` containing at least :code:`{}` **must** be present. The search order for the config file is as follows:

1. the folder of the file that is currently running (or the current working directory if not running from a file)
2. the parent folder of the folder from the previous folder
3. any parent folder named :code:`hornero-event-classifier` from the original folder

See :py:class:`Config` for config options. Custom variables may also be set and retrieved using :code:`CONFIG["custom_var"]`.
Variables can use :code:`~/` at the start of a path to reference their home directory. File path variable can reference one
another as well as the folder of the config file (using the keyword :code:`config_folder`) as shown below::

    {
        "some_var": "{config_folder}/../..",
        "data_root": "{some_var}/data",
        "yolo_path": "{data_root}/filename.csv",
        "boris_path": "~/Documents/boris.csv"
    }

.. warning::
    Some parts of :mod:`hornero_event_classifier` require video metadata info such as frame size or video length. As such it
    is required that at least one variable of :py:meth:`Config.video_metadata`, :py:meth:`Config.video_root`, or
    :py:meth:`Config.default_video_metadata` must be set or the defaults must be valid. However, the use of
    :py:meth:`Config.default_video_metadata` is strongly discouraged and has no default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import __main__

if TYPE_CHECKING:
    from hornero_event_classifier.tools.video import VideoMetadata

# defaults:
# required name of config files
CONFIG_FILE_NAME: str = "hec-config.json"
# directory to search for in current file path to find fallback config file
LIBRARY_FOLDER: str = "hornero-event-classifier"
# value to place into the config if not already set
CONFIG_DEFAULTS: dict[str, Any] = {
    "data_root": "{config_folder}/data",
    "yolo_path": "{data_root}/YOLO",
    "results_path": "{data_root}/hec_output.csv",
    "boris_path": "{data_root}/DB_BORIS.csv",
    "video_metadata": "{data_root}/video_metadata.json",
    "video_root": "~/Videos/videos_BORIS",
    "auto_gen_video_metadata": True,
}


class Config:
    """The :code:`Config` class holds a configuration dictionary and allows to variables through properties and indexing while not
    allowing for writing of variables. Variables can also be retrieved using indexing (:code:`Config['data_root`]`) allowing for
    the retrieving of custom config variables. This class should not be initiated by the user or any other modules.

    :param data: a dictionary of config variables
    :type data: dict[str, Any]
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def data_root(self) -> Path:
        """The root directory within which other paths can be placed in reference to. The default is: :code:`{config_folder}/data`

        .. note:: this config variable is not used internally and is just for personal convenience

        :rtype: Path
        """
        return self._data["data_root"]

    @property
    def yolo_path(self) -> Path:
        """The directory where raw input yolo csv files can be found. The default is: :code:`{data_root}/YOLO`

        .. note:: this config variable is not used internally and is just for personal convenience

        :rtype: Path
        """
        return self._data["yolo_path"]

    @property
    def results_path(self) -> Path:
        """The path to save classifications results to. The default is: :code:`{data_root}/hec_output.csv`

        .. note:: this config variable is not used internally and is just for personal convenience

        :rtype: Path
        """
        return self._data["results_path"]

    @property
    def boris_path(self) -> Path:
        """The path to a boris validation csv file. The default is: :code:`{data_root}/DB_BORIS.csv`

        .. note:: this config variable is not used internally and is just for personal convenience

        :rtype: Path
        """
        return self._data["boris_path"]

    @property
    def video_metadata(self) -> Path:
        """The path to a video metadata json file. If this is not set or the file can not be found, metadata will be extracted
        directly from video in :py:meth:`Config.video_root`. The default is: :code:`{data_root}/video_metadata.json`

        :rtype: Path
        """
        return self._data["video_metadata"]

    @property
    def video_root(self) -> Path:
        """The path to a directory of the original videos. The video directory must be structured such that videos are placed
        into their nested subdirectories (e.g. :code:`{video_root}/n1/n1_....mp4`, :code:`{video_root}/n8/n8_....mp4`). The default
        is: :code:`~/Videos/videos_BORIS`

        .. warning:: This variable is required to use :py:mod:`animate` functionality

        .. warning::
            If :py:meth:`Config.video_metadata` and :py:meth:`Config.video_root` are not set then the system falls back
            to :py:meth:`Config.default_video_metadata` which is not recommended

        :rtype: Path
        """
        return self._data["video_root"]

    @property
    def auto_gen_video_metadata(self) -> bool:
        """A boolean value if video metadata should be generated automatically. The default is: :code:`True`

        :rtype: bool
        """
        return self._data["auto_gen_video_metadata"]

    @property
    def default_video_metadata(self) -> VideoMetadata:
        """A default video metadata dictionary to be used if :py:meth:`Config.video_metadata` and :py:meth:`Config.video_root` are
        not set. The dictionary keys should include:

        +------------+-------+---------------------------------+
        | Key        | Type  | Meaning                         |
        +============+=======+=================================+
        | fps        | str   | frames per second (as fraction) |
        +------------+-------+---------------------------------+
        | duration_s | float | duration of video in seconds    |
        +------------+-------+---------------------------------+
        | duration_f | int   | duration of video in frames     |
        +------------+-------+---------------------------------+
        | width      | int   | the width of video frames       |
        +------------+-------+---------------------------------+
        | height     | int   | the height of video frames      |
        +------------+-------+---------------------------------+

        .. code-block:: python3

            {
                "fps": "30/1",
                "duration_s": 3600.0,
                "duration_f": 108000,
                "width": 1280,
                "height": 720,
            }

        There is no default value.

        :raises AttributeError: raised if default_video_metadata is called but not set. There is no default value.
        :rtype: VideoMetadata
        """
        if "default_video_metadata" not in self._data:
            raise AttributeError("default_video_metadata must be set explicitly")
        return self._data["default_video_metadata"]

    def __getitem__(self, key):
        return self._data[key]


def _find_config_file():
    # get path of running file if using script
    main_path: str | None = getattr(__main__, "__file__", None)
    path: Path
    # if not running a script default to current working directory.
    if main_path is None:
        path = Path.cwd()
    else:
        path = Path(main_path)
    # check if config file exists in current directory
    if (path / CONFIG_FILE_NAME).is_file():
        out = path / CONFIG_FILE_NAME
    # check if config file exists one folder up
    elif (path / ".." / CONFIG_FILE_NAME).is_file():
        out = path / ".." / CONFIG_FILE_NAME
    # try to find LIBRARY_FOLDER in current path
    else:
        full_path: str = path.resolve().as_posix()
        cut_pos: int = full_path.find(LIBRARY_FOLDER)
        # if folder found set path to default config file
        if cut_pos != -1:
            out = Path(full_path[: cut_pos + len(LIBRARY_FOLDER)]) / CONFIG_FILE_NAME
        # if file not found give up and raise error
        if cut_pos == -1 or not out.is_file():
            raise FileNotFoundError(f"No config file ({CONFIG_FILE_NAME}) found")
    return out.resolve()


def _load_config(path: Path) -> Config:
    # load config file
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # add parent folder path of config file
    data["config_folder"] = path.parent
    # check for unset arguments and replace with defaults
    for key, value in CONFIG_DEFAULTS.items():
        if key not in data:
            data[key] = value
    # resolve references to other config variables
    for key, value in data.items():
        if isinstance(value, str):
            value = value.format(**data)
            if value.startswith("~/"):
                data[key] = Path.home() / value[2:]
            else:
                data[key] = Path(value)
    # return config instance
    return Config(data)


#: the loaded instance of :py:class:`Config` that other modules can reference
CONFIG: Config = _load_config(_find_config_file())

if __name__ == "__main__":
    print(CONFIG)
