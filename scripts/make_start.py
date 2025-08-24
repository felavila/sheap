#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate getting_started.rst from README.rst:

1) Find the first real section (title + underline).
2) Replace it with an overline+title+underline using '=':
       ============
       Welcome ...
       ============
3) For all subsequent section underlines made of '=', convert them to '-'.
4) Any other repeated-punctuation underline (~, ^, -) is normalized to '-'.

Note: By default we start from the first section onward (omitting any preface like a logo).
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
    Find index i where lines[i] is a title and lines[i+1] is its underline.
    Return 0 if not found.
    """
    for i in range(len(lines) - 1):
        title = lines[i].rstrip("\n")
        underline = lines[i + 1].rstrip("\n")
        if title.strip() and is_underline(underline) and len(underline) >= len(title):
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

    new_body: list[str] = []
    first_replaced = False
    i = 0

    while i < len(body_lines):
        # Replace the *first* section with overline + title + underline of '='
        if (
            not first_replaced
            and i + 1 < len(body_lines)
            and is_underline(body_lines[i + 1].rstrip("\n"))
        ):
            title_len = len(new_title)
            over = "=" * title_len + "\n"
            title_line = f"{new_title}\n"
            under = "=" * title_len + "\n"
            new_body.extend([over, title_line, under])
            first_replaced = True
            i += 2  # skip original title + underline
            continue

        line = body_lines[i]

        # For all other underline lines:
        if is_underline(line.rstrip("\n")):
            ch = line.strip()[0] if line.strip() else "-"
            # Convert '=' (and everything else) to '-' of the same length
            # (this meets "replace the ==== for --------" for non-Welcome titles)
            new_body.append("-" * len(line.rstrip("\n")) + "\n")
            i += 1
            continue

        # Otherwise, copy through
        new_body.append(line)
        i += 1

    out_path.write_text("".join(new_body), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate getting_started.rst from README.rst")
    p.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("README.rst"),
        help="Path to README.rst (default: README.rst)",
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
        help="Replacement title text for the first section",
    )
    args = p.parse_args()

    rewrite_readme(args.input, args.output, new_title=args.title)


if __name__ == "__main__":
    main()
