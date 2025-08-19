#!/usr/bin/env python3
"""
Search recursively for f-strings that use double quotes
(e.g. f"..." or f\"\"\"...\"\"\").
"""

import re
from pathlib import Path
import argparse

# Regex to match f-strings starting with f" or f"""
FSTRING_PATTERN = re.compile(r'''(?<!\w)f"{1,3}''')

def find_fstrings(root: Path, include_tests: bool = False):
    for py_file in root.rglob("*.py"):
        pstr = str(py_file)
        # Skip common junk folders
        if any(seg in pstr for seg in ("/.venv/", "/venv/", "/.tox/", "/.mypy_cache/", "/.git/", "/build/", "/dist/")):
            continue
        if not include_tests and "/tests/" in pstr:
            continue

        try:
            text = py_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[skip] {py_file}: {e}")
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if FSTRING_PATTERN.search(line):
                yield py_file, lineno, line.strip()


def main():
    ap = argparse.ArgumentParser(description="Find f-strings with double quotes recursively")
    ap.add_argument("root", nargs="?", default=".", help="Root directory (default: .)")
    ap.add_argument("--include-tests", action="store_true", help="Also scan tests/ (default: skip)")
    args = ap.parse_args()

    root = Path(args.root)
    found = False
    for path, lineno, line in find_fstrings(root, args.include_tests):
        found = True
        print(f"{path}:{lineno}: {line}")

    if not found:
        print("✅ No f\"...\" strings found.")


if __name__ == "__main__":
    main()
