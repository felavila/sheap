import yaml
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT     = PROJECT_ROOT / "sheap" / "SuportData"
DOCS_ROOT    = PROJECT_ROOT / "docs" / "source" / "SuportData"
DOCS_ROOT.mkdir(exist_ok=True, parents=True)

supportdata_index = [
    "SuportData Resources",
    "=" * 22,
    "",
    ".. toctree::",
    "   :maxdepth: 1",
    ""
]

# Skip undesired dirs
SKIP_DIRS = {"__pycache__"}

for subdir in sorted(SRC_ROOT.iterdir()):
    if not subdir.is_dir() or subdir.name in SKIP_DIRS:
        continue

    section = subdir.name
    print(f"📁 Processing {section}")
    target_dir = DOCS_ROOT / section
    target_dir.mkdir(parents=True, exist_ok=True)

    section_index = [f"{section.title()}", "=" * len(section), ""]

    # Include README.md if present
    readme_path = subdir / "README.md"
    if readme_path.exists():
        section_index += [
            f".. include:: ../../../../sheap/SuportData/{section}/README.md",
            "   :parser: myst_parser.sphinx_",
            ""
        ]

    # Only LineRepository gets YAML rendering
    if section == "LineRepository":
        yaml_dir = target_dir / "yaml"
        yaml_dir.mkdir(exist_ok=True)
        has_yaml = False

        for file in sorted(subdir.glob("*.yaml")):
            try:
                data = yaml.safe_load(file.read_text())
            except Exception as e:
                print(f"⚠️  Failed to parse {file.name}: {e}")
                continue

            # Support both top-level list and {"region": [...]}
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and "region" in data:
                rows = data["region"]
            else:
                print(f"⚠️  Skipping {file.name}: not a valid region list or format")
                continue

            if not rows:
                continue

            # Determine full set of keys and max widths
            keys = sorted({k for row in rows for k in row.keys()})
            col_widths = {k: max(len(k), max((len(str(row.get(k, ""))) for row in rows))) for k in keys}

            def format_row(row_dict):
                return "| " + " | ".join(f"{str(row_dict.get(k, '')).ljust(col_widths[k])}" for k in keys) + " |"

            divider = "+-" + "-+-".join("-" * col_widths[k] for k in keys) + "-+"
            header = "| " + " | ".join(k.ljust(col_widths[k]) for k in keys) + " |"
            header_sep = "+=" + "=+=".join("=" * col_widths[k] for k in keys) + "=+"

            rst_file = yaml_dir / f"{file.stem}.rst"
            with open(rst_file, "w") as f:
                f.write(f"{file.stem.title()} \n")
                f.write("=" * (len(file.stem) + 8) + "\n\n")
                f.write(divider + "\n")
                f.write(header + "\n")
                f.write(header_sep + "\n")
                for row in rows:
                    f.write(format_row(row) + "\n")
                    f.write(divider + "\n")

            has_yaml = True

        if has_yaml:
            section_index += [
                "YAML Files",
                "----------",
                "",
                ".. toctree::",
                "   :maxdepth: 1",
                "   :glob:",
                "",
                "   yaml/*",
                ""
            ]

    # Write section-level index.rst
    with open(target_dir / "index.rst", "w") as f:
        f.write("\n".join(section_index))

    # Add to global TOC
    supportdata_index.append(f"   SuportData/{section}/index")

# Write top-level supportdata.rst
with open(PROJECT_ROOT / "docs" / "source" / "supportdata.rst", "w") as f:
    f.write("\n".join(supportdata_index))

#print("✅ All SuportData documentation has been generated.")
