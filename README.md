# Getting Started

> [!IMPORTANT] 
> There are several steps that need to be taken before you can start classifying YOLO results.Read this document once 
> before you begin.

## Installation

The Hornero Event Classifier is designed to be used from the repository root. Installing in editable mode is recommended
so you can modify code while the package remains available.

```bash
conda env create -f conda_environment.yaml
conda activate hec_venv
pip install -e .
```

If you prefer pip-only installation, run:

```bash
pip install -e .
```

> [!WARNING]
> If you do not use conda you may need to install `ffmpeg` yourself, if not already installed.
    
### Building documentation

The documentation is built on demand. Use the included script to build the docs and open them in your browser:

```bash
python run/docs.py
```

## Required data

The classifier requires at minimum:

- YOLO detection CSV files in `data/YOLO/`
- a generated metadata repository at `data/video_metadata.json`

The repository does not include these directories or files by default, so create `data/` and place your YOLO CSV
files under `data/YOLO/` before running the pipeline.

### Video metadata

Most scripts depend on `data/video_metadata.json`. Generate this file from your YOLO CSV files and the corresponding
video files with:

```bash
python run/gen_metadata_file.py
```

By default, `run/gen_metadata_file.py` expects videos to be located at:

`~/Videos/videos_BORIS/<nest>/<video_id>.mp4`

After metadata generation, video files are only required for the animation pipeline.
If your videos are stored in a different location, update `VIDEOS_ROOT_PATH` in `run/defaults.py`.

## Pipeline overview

The repository contains a small pipeline of helper scripts and a core package. The main entry points are:

- `run/gen_metadata_file.py`: generate `data/video_metadata.json` from YOLO files and video files
- `run/classify.py`: classify videos and write results to `data/hec_output.csv`
- `run/validate.py`: validate classification output against BORIS ground truth
- `run/animate.py`: animate a single classified video scene
- `run/get_weights.py`: estimate metric weights from validation data

The default classifier is loaded from `run/weights.json` and uses a threshold-based scoring pipeline.

### Classifying videos

With metadata ready, classify all videos with the default classifier and save results to `data/hec_output.csv`:

```bash
python run/classify.py
```

Useful options:

- `--restart`: re-classify all videos and overwrite existing `data/hec_output.csv`
- `--no-progress`: suppress progress messages
- `--no-plot`: skip the interactive results plot
- `--max-bird-gap <frames>`: maximum gap size when splitting bird detections
- `--no-fill` / `--fill-at-edge`: control gap filling for missing bounding boxes
- `--remove-low-conf <score>`: confidence threshold for bird detections
- `--combine-events-within <frames>`: merge nearby events into a single event
- `--min-event-len <frames>`: discard events shorter than this length

### Validation

If you have BORIS ground truth annotations in `data/DB_BORIS.csv`, validate the classification output with:

```bash
python run/validate.py
```

Key validation options:

- `--overlap <ratio>`: minimum overlap required for a true positive (default: 0.8)
- `--print-long`: print per-video statistics
- `--no-print`: suppress printed output
- `--no-plot`: disable the validation plot
- `--white-list` / `--black-list`: include or exclude video IDs by prefix

### Animating results

View a single video and its detected events with:

```bash
python run/animate.py <video_id>
```

Example:

```bash
python run/animate.py n10_d4_c1_1_cl2
```

Animation options:

- `--scale`: rescale the displayed video
- `--frame`: starting frame for playback
- `--clip start,end`: limit playback to a frame interval
- `--auto-play`: automatically start playback
- `--save <path>`: save the animation to a video file

### Estimating classifier weights

Use `run/get_weights.py` to estimate weights for the threshold classifier from validation data.
It applies a generalized linear model to selected metrics and provides a starting point for weight tuning.

.. code-block:: console

    python run/get_weights.py

If no metrics are provided, the script uses all available metrics.

### Customizing defaults

Default file paths are configured in `run/defaults.py`. Update `VIDEOS_ROOT_PATH` or other constants there if your
local dataset layout differs from the repository defaults.