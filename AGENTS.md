# Repository Guidelines

## Project Structure & Module Organization
Core package code lives in `src/hornero_event_classifier/`. Use `core/` for scene, metadata, and shared data structures, `classifiers/` for scoring logic, and `tools/` for plotting, animation, and validation helpers. Repository-level workflow scripts live in `run/` and are the main local entry points. Sphinx docs are under `docs/source/`. Runtime data is expected under `data/` and is not committed; the pipeline reads from `data/YOLO/` and `data/video_metadata.json` and writes outputs such as `data/hec_output.csv`.

## Build, Test, and Development Commands
Set up the environment from the repository root:

```bash
conda env create -f conda_environment.yaml
conda activate hec_venv
pip install -e .
```

Common workflows:

```bash
python run/gen_metadata_file.py   # build data/video_metadata.json
python run/classify.py            # classify all videos
python run/validate.py            # compare output with BORIS annotations
python run/animate.py <video_id>  # inspect one scene visually
python run/docs.py --rebuild      # rebuild Sphinx docs
python -m py_compile src/hornero_event_classifier/**/*.py run/*.py
```

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for modules/functions/variables, PascalCase for classes, and explicit type hints on public APIs. Match the repository’s preference for short module docstrings and focused function docstrings where behavior is not obvious. Keep new scripts in `run/` named by action, for example `run/export_metrics.py`. `pylint` is available in the environment; use it when changing non-trivial logic.

## Testing Guidelines
There is no dedicated `tests/` suite yet, so validate changes through the pipeline. At minimum, run `python run/classify.py --no-plot` on representative data, then `python run/validate.py --no-plot` if `data/DB_BORIS.csv` is available. Use `python run/animate.py <video_id>` for spot checks on event timing and overlays. Do not add a new testing architecture for this repository unless it is explicitly requested.

## Commit & Pull Request Guidelines
Recent commits use short, direct summaries such as `documentation improvements` and `windows compatibility update`. Keep commit titles concise, lower-noise, and scoped to one change. Pull requests should describe the affected pipeline stage, note any required data/config changes (`run/defaults.py`, `hec-config.json`), and include screenshots when UI or plot output changes.

## Data & Configuration Notes
Do not hardcode machine-specific paths outside the existing config points. Adjust local dataset locations in `run/defaults.py` or `hec-config.json`, and keep generated data files out of version control unless explicitly requested.
