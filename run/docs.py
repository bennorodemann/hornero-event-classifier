from __future__ import annotations

import argparse
import re
import subprocess
import webbrowser
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_project_version() -> str:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
        import tomli as tomllib  # type: ignore

    pyproject_path = _repo_root() / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return data["project"]["version"]


def _read_built_version(index_path: Path) -> str | None:
    if not index_path.exists():
        return None
    html = index_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Documentation version:\\s*([0-9A-Za-z.+-]+)", html)
    return match.group(1) if match else None


def _build_docs(sphinx_build: str, source_dir: Path, build_dir: Path) -> int:
    result = subprocess.run(
        [sphinx_build, "-M", "html", str(source_dir), str(build_dir)],
        check=False,
    )
    return result.returncode


def _should_rebuild(reason: str, force: bool) -> bool:
    if force:
        return True
    answer = input(f"{reason} Rebuild docs now? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Open docs, rebuilding when the build is missing or out of date.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild docs automatically when needed (no prompt).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the docs in a browser after checks.",
    )
    parser.add_argument(
        "--sphinx-build",
        default="sphinx-build",
        help="Path to the sphinx-build executable.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    source_dir = repo_root / "docs" / "source"
    build_dir = repo_root / "docs" / "build"
    index_path = build_dir / "html" / "index.html"

    project_version = _read_project_version()
    built_version = _read_built_version(index_path)

    if built_version is None:
        if _should_rebuild("Docs are not built yet.", args.rebuild):
            if _build_docs(args.sphinx_build, source_dir, build_dir) != 0:
                return 1
    elif built_version != project_version:
        reason = "Docs are out of date " f"(built {built_version}, project {project_version})."
        if _should_rebuild(reason, args.rebuild):
            if _build_docs(args.sphinx_build, source_dir, build_dir) != 0:
                return 1
    else:
        print(f"Docs are up to date (version {built_version}).")

    if not args.no_open:
        if not index_path.exists():
            print("Docs index not found after build.")
            return 1
        webbrowser.open(index_path.as_uri(), new=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
