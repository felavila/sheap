#!/usr/bin/env python3
"""
Generate an .rst page listing all available continuum profiles from
`sheap.Profile.profiles_continuum`, including signatures, param order,
and docstrings, ready for Sphinx.

Usage:
  python tools/make_available_continuum_profiles.py \
      --module sheap.Profile.profiles_continuum \
      --out docs/source/api/available_continuum_profiles.rst
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import textwrap
from pathlib import Path
from typing import Iterable, Tuple


def underline(title: str, ch: str = "=") -> str:
    return f"{title}\n{ch * len(title)}\n\n"


def collect_functions(mod) -> Iterable[Tuple[str, object]]:
    """Return iterable of (name, obj) for callables to document."""
    # Prefer __all__ to preserve your chosen order.
    names = getattr(mod, "__all__", None)
    items = []

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

        # Stable sort by name if no __all__ is present
        items.sort(key=lambda x: x[0])

    return items


def render_page(module_qualname: str, out_path: Path) -> None:
    mod = importlib.import_module(module_qualname)
    funcs = list(collect_functions(mod))

    title = "Available Continuum Profiles"
    rst = []
    rst.append(underline(title, "="))
    rst.append(
        f"This page is auto-generated from :py:mod:`{module_qualname}`.\n\n"
    )

    # Optional global note if delta0 exists
    if hasattr(mod, "delta0"):
        rst.append(f".. note:: Normalization wavelength ``delta0 = {getattr(mod, 'delta0')}`` Å.\n\n")

    # Quick list
    rst.append("Profiles\n--------\n\n")
    for name, _ in funcs:
        rst.append(f"* :py:func:`{module_qualname}.{name}`\n")
    rst.append("\n")

    # Detailed sections
    for name, func in funcs:
        # Section header
        rst.append(underline(name, "^"))

        # Function signature (best-effort)
        try:
            sig = str(inspect.signature(func))
        except (TypeError, ValueError):
            sig = "(...)"

        rst.append(f".. py:function:: {module_qualname}.{name}{sig}\n\n")

        # Param order if provided by your decorator
        param_names = getattr(func, "param_names", None)
        if param_names:
            rst.append(f"   **Parameter order (SHEAP):** ``{', '.join(param_names)}``\n\n")

        # Docstring content, indented under the directive
        doc = inspect.getdoc(func) or "No docstring available."
        rst.append(textwrap.indent(doc, "   ") + "\n\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(rst), encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", default="sheap.Profile.profiles_continuum")
    p.add_argument("--out", default="docs/source/available_continuum_profiles.rst")
    args = p.parse_args()
    render_page(args.module, Path(args.out))


if __name__ == "__main__":
    main()
