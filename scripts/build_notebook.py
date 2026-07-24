"""
One-off helper used to author the notebooks/ teaching notebooks. Not part
of the runtime pipeline -- run manually when adding/regenerating a notebook.
Takes a list of (type, content) tuples and writes valid nbformat v4 JSON.
"""

from __future__ import annotations

import json
import sys


def make_notebook(cells: list[tuple[str, str]], out_path: str) -> None:
    nb_cells = []
    for cell_type, source in cells:
        lines = source.split("\n")
        src_list = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": src_list,
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Wrote {out_path}")
