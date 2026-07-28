"""
One-off helper used to author the notebooks/ teaching notebooks. Not part
of the runtime pipeline -- run manually when adding/regenerating a notebook.
Takes a list of (type, content) tuples and writes valid nbformat v4 JSON.
"""

from __future__ import annotations

import json
import sys


def make_notebook(cells: list[tuple[str, str]], out_path: str) -> None:
    # cells: a flat list of (cell_type, source) tuples, e.g. ("markdown", "...")
    # or ("code", "..."). Each gen_XX script builds one of these lists as a
    # literal in-order script for the notebook it authors.
    nb_cells = []
    for cell_type, source in cells:
        # nbformat stores a cell's source as a LIST of lines, not one big
        # string, and every line except the last must keep its trailing "\n"
        # (the last line only gets one if it wasn't empty to begin with) --
        # this is what Jupyter itself writes, so splitting/rejoining here
        # keeps the output byte-for-byte compatible with a notebook saved
        # from the Jupyter UI.
        lines = source.split("\n")
        src_list = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": src_list,
        }
        if cell_type == "code":
            # Code cells need these two extra keys (even when empty/None) or
            # Jupyter/nbformat will reject the file as malformed -- markdown
            # cells don't have an execution count or outputs at all.
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    # Minimal valid nbformat v4.5 document: the cell list above plus the
    # kernel/language metadata Jupyter needs to know how to open and run it.
    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    # indent=1 keeps the .ipynb JSON diff-friendly in git (one key per line)
    # rather than a single minified line, at the cost of a larger file.
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Wrote {out_path}")
