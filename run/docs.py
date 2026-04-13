"""
Documentation management script.

This script handles building and opening the project documentation using Sphinx. It checks if the documentation is built
yet, and will automatically build it if necessary, then opens the documentation in a web browser.
"""

from __future__ import annotations

import subprocess
import webbrowser
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
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

args = parser.parse_args()

# get docs root directory
docs_dir = Path(__file__).parent.parent / "docs"

# path to docs homepage
target = docs_dir / "build/html/index.html"

# if rebuild is requested or target does not exist: build docs
if args.rebuild or not target.exists():
    result = subprocess.run(
        ["sphinx-build", "-M", "html", str(docs_dir / "source"), str(docs_dir / "build")],
        check=False,
    )

# open docs in web browser
if not args.no_open:
    if target.exists():
        webbrowser.open(target.as_uri(), new=2)
    else:
        print("Docs index not found after build.")
