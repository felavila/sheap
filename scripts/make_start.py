#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a "getting_started.rst" from a README.rst by:
  1) Finding the first real section (title + underline).
  2) Replacing that first title with "Welcome to sheap’s documentation"
     and an '=' underline of matching length.
  3) Converting any underline lines of repeated punctuation to '-',
     EXCEPT lines made of '=' which are preserved verbatim.

This assumes the README has a section like:

Spectral Handling and Estimation of AGN Parameters
==================================================

…and possibly many other sections that (in your current README) also use '='.
We keep all '====...' lines unchanged, per your request.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


UNDERLINE_RE = re.compile(r"^([=~^\-])\1+$")


def is_underline(line: str) -> bool:
    """Return True if the line is a repeated punctuation underline."""
    return bool(UNDERLINE_RE.match(line.rstrip("\n")))


def find_first_section_start(lines: list[str]) -> int:
    """
    Find the index i such that lines[i] is a title and lines[i+1] is its underline.
    Returns the index of the title line (i). If not found, returns 0.
    """
    for i in range(len(lines) - 1):
        title = lines[i].rstrip("\n")
        underline = lines[i + 1].rstrip("\n")
        if (
            title.strip() != ""  # title not empty
            and is_underline(underline)
            and len(underline) >= len(title)
        ):
            return i
    return 0


def rewrite_readme(
    readme_path: Path,
    out_path: Path,
    new_title: str = "Welcome to sheap’s documentation",
) -> None:
    text = readme_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # 1) Locate first section start
    start = find_first_section_start(lines)

    # 2) Work from that section onward
    body_lines = lines[start:]

    # 3) Build new body with first section swapped; keep '=' underlines intact; convert others to '-'
    new_body: list[str] = []
    first_replaced = False
    i = 0

    while i < len(body_lines):
        line = body_lines[i]
        # Handle the first section: replace its title and underline
        if (
            not first_replaced
            and i + 1 < len(body_lines)
            and is_underline(body_lines[i + 1].rstrip("\n"))
        ):
            # Replace title + underline with the new heading (always '=' underline)
            title_line = f"{new_title}\n"
            underline_line = "=" * len(new_title) + "\n"
            new_body.append(title_line)
            new_body.append(underline_line)
            first_replaced = True
            i += 2
            continue

        # If this line is a pure underline of repeated punctuation
        if is_underline(line.rstrip("\n")):
            ch = line.strip()[0] if line.strip() else "-"
            if ch == "=":
                # Keep all '====...' lines verbatim (do not touch)
                new_body.append(line)
            else:
                # Convert other underline styles to '-' of the same length
                new_body.append("-" * len(line.rstrip("\n")) + "\n")
            i += 1
            continue

        # Otherwise, just copy the line
        new_body.append(line)
        i += 1

    # 4) Write output file
    out_path.write_text("".join(new_body), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate getting_started.rst from README.rst")
    p.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("README.rst"),
        help="Path to README.rst (default: ../../README.rst)",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("docs/source/getting_started.rst"),
        help="Output .rst path (default: getting_started.rst)",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Welcome to sheap’s documentation",
        help="Replacement title for the first section",
    )
    args = p.parse_args()

    rewrite_readme(args.input, args.output, new_title=args.title)


if __name__ == "__main__":
    main()
