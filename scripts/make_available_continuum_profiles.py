#!/usr/bin/env python3
"""
Generate .rst pages for multiple modules that expose profile functions.

Defaults:
  --modules sheap.Profiles.profiles_continuum sheap.Profiles.profiles_lines
  --outdir  docs/source

Examples
--------
# Run with defaults (continuum + lines into docs/source/)
python tools/make_available_profiles_pages.py

# Custom rename for outputs
python tools/make_available_profiles_pages.py \
  --rename profiles_continuum=available_continuum_profiles.rst \
  --rename profiles_lines=available_line_profiles.rst
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List, Tuple, Dict


DEFAULT_MODULES = [
    "sheap.Profiles.profiles_continuum",
    "sheap.Profiles.profile_lines",
]


def underline(title: str, ch: str = "=") -> str:
    return f"{title}\n{ch * len(title)}\n\n"


def collect_functions(mod) -> Iterable[Tuple[str, object]]:
    """Return iterable of (name, obj) for callables to document."""
    names = getattr(mod, "__all__", None)
    items: List[Tuple[str, object]] = []

    if names:
        for n in names:
            obj = getattr(mod, n, None)
            if callable(obj):
                items.append((n, obj))
    else:
        for n, obj in vars(mod).items():
            if n.startswith("_"):
                continue
            if callable(obj) and getattr(obj, "__module__", "") == mod.__name__:
                items.append((n, obj))
        items.sort(key=lambda x: x[0])

    return items


def render_single_module(module_qualname: str) -> str:
    """Return the full .rst text for one module."""
    mod = importlib.import_module(module_qualname)
    funcs = list(collect_functions(mod))

    title = "Available " + module_qualname.split(".")[-1].replace("_", " ").title()
    rst: List[str] = []
    rst.append(underline(title, "="))
    rst.append(f"This page is auto-generated from :py:mod:`{module_qualname}`.\n\n")

    # Optional global notes (example: continuum's delta0)
    if hasattr(mod, "delta0"):
        rst.append(f".. note:: Normalization wavelength ``delta0 = {getattr(mod, 'delta0')}`` Å.\n\n")

    # Quick index
    rst.append("Profiles\n--------\n\n")
    if not funcs:
        rst.append("* *(No functions found)*\n\n")
        return "".join(rst)
    for name, _ in funcs:
        rst.append(f"* :py:func:`{module_qualname}.{name}`\n")
    rst.append("\n")

    # Detailed sections per function
    for name, func in funcs:
        rst.append(underline(name, "^"))
        try:
            sig = str(inspect.signature(func))
        except (TypeError, ValueError):
            sig = "(...)"
        rst.append(f".. py:function:: {module_qualname}.{name}{sig}\n\n")

        param_names = getattr(func, "param_names", None)
        if param_names:
            rst.append(f"   **Parameter order (sheap):** ``{', '.join(param_names)}``\n\n")

        doc = inspect.getdoc(func) or "No docstring available."
        rst.append(textwrap.indent(doc, "   ") + "\n\n")

    return "".join(rst)


def compute_outfile(module_qualname: str, outdir: Path, renames: Dict[str, str]) -> Path:
    """
    Determine output filename.
    - If the module's last segment appears in renames, use that filename.
    - Else use 'available_{last}.rst'
    """
    last = module_qualname.split(".")[-1]
    print(last)
    fname = renames.get(last, f"available_{last}.rst")
    return outdir / fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--modules",
        nargs="+",
        default=DEFAULT_MODULES,
        help="One or more module paths. Defaults to: %(default)s",
    )
    ap.add_argument(
        "--outdir",
        default="docs/source",
        help="Directory to write the .rst files into (created if missing).",
    )
    ap.add_argument(
        "--rename",
        action="append",
        default=[],
        help="Optional per-module last-segment filename override like "
             "'profiles_continuum=available_continuum_profiles.rst'. "
             "May be given multiple times.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse --rename pairs into a dict
    renames: Dict[str, str] = {}
    for pair in args.rename:
        if "=" not in pair:
            print(f"[WARN] Ignoring malformed --rename '{pair}' (expected key=filename.rst)", file=sys.stderr)
            continue
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key or not val:
            print(f"[WARN] Ignoring malformed --rename '{pair}'", file=sys.stderr)
            continue
        renames[key] = val

    rc = 0
    for modname in args.modules:
        try:
            rst_text = render_single_module(modname)
        except Exception as e:
            rc = 1
            print(f"[ERROR] Failed to render {modname}: {e}", file=sys.stderr)
            continue

        outpath = compute_outfile(modname, outdir, renames)
        try:
            outpath.write_text(rst_text, encoding="utf-8")
            print(f"Wrote {outpath}")
        except Exception as e:
            rc = 1
            print(f"[ERROR] Failed to write {outpath}: {e}", file=sys.stderr)

    sys.exit(rc)


if __name__ == "__main__":
    main()
