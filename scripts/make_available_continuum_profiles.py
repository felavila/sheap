#!/usr/bin/env python3
"""
Generate simple .rst pages with a custom title + automodule directive.

Defaults:
  --modules sheap.Profiles.profiles_continuum sheap.Profiles.profile_lines
  --outdir  docs/source

Examples
--------
# Run with defaults
python tools/make_available_profiles_pages.py

# Custom file name + custom title
python tools/make_available_profiles_pages.py \
  --rename profile_lines=available_line_profiles.rst \
  --title  profile_lines="Available line profiles"
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

DEFAULT_MODULES = [
    "sheap.Profiles.profiles_continuum",
    "sheap.Profiles.profile_lines",   # note: singular 'profile_lines' here
]

def underline(title: str, ch: str = "=") -> str:
    return f"{title}\n{ch * len(title)}\n\n"

def default_title(mod: str) -> str:
    last = mod.split(".")[-1]
    if last == "profiles_continuum":
        return "Available continuum profiles"
    if last in ("profile_lines", "profiles_lines"):
        return "Available line profiles"
    return "Available " + last.replace("_", " ")

def render_page(module_qualname: str, custom_title: str | None = None) -> str:
    title = custom_title or default_title(module_qualname)
    rst: List[str] = []
    rst.append(underline(title, "="))
    rst.append(f".. automodule:: {module_qualname}\n")
    rst.append("   :members:\n")
    rst.append("   :undoc-members:\n")
    rst.append("   :show-inheritance:\n\n")
    return "".join(rst)

def compute_outfile(module_qualname: str, outdir: Path, renames: Dict[str, str]) -> Path:
    last = module_qualname.split(".")[-1]
    fname = renames.get(last, f"available_{last}.rst")
    return outdir / fname

def parse_kv_list(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pair in items:
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="+", default=DEFAULT_MODULES)
    ap.add_argument("--outdir", default="docs/source")
    ap.add_argument("--rename", action="append", default=[],
                    help="Override output filename: e.g. 'profile_lines=available_line_profiles.rst'")
    ap.add_argument("--title", action="append", default=[],
                    help="Override page title: e.g. 'profile_lines=Available line profiles'")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    renames = parse_kv_list(args.rename)
    titles  = parse_kv_list(args.title)

    rc = 0
    for mod in args.modules:
        try:
            last = mod.split(".")[-1]
            rst_text = render_page(mod, titles.get(last))
            outpath = compute_outfile(mod, outdir, renames)
            outpath.write_text(rst_text, encoding="utf-8")
            print(f"Wrote {outpath}")
        except Exception as e:
            rc = 1
            print(f"[ERROR] Failed for {mod}: {e}")
    raise SystemExit(rc)

if __name__ == "__main__":
    main()
