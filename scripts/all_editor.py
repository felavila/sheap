import os
from pathlib import Path
import re

# === Customizable ===
EXCLUDE = {"__pycache__", "__main__", "_logging", "tests", "scripts", "SuportData","Core"}
ROOT_PACKAGE = Path(__file__).resolve().parent.parent / "sheap"  # adjust if needed
print(ROOT_PACKAGE)
# === Helpers ===

def generate_public_entries(pkg_path: Path) -> list[str]:
    entries = []
    for item in sorted(pkg_path.iterdir()):
        name = item.stem if item.is_file() else item.name
        if name.startswith("_") or name in EXCLUDE:
            continue
        if item.is_file() and item.suffix == ".py":
            entries.append(name)
        elif item.is_dir() and (item / "__init__.py").exists():
            entries.append(name)
    return sorted(entries)

def replace_or_append_all(init_path: Path, entries: list[str]):
    new_all_block = "__all__ = [\n" + "\n".join(f'    "{e}",' for e in entries) + "\n]\n"

    if init_path.exists():
        content = init_path.read_text()

        # Replace existing __all__ block
        if "__all__" in content:
            updated = re.sub(r"(?s)__all__\s*=\s*\[.*?\]", new_all_block.strip(), content)
        else:
            updated = content.strip() + "\n\n" + new_all_block
    else:
        updated = "# Auto-generated __init__.py\n\n" + new_all_block

    init_path.write_text(updated + "\n")

def process_package_recursively(pkg_root: Path):
    for current_path, dirs, files in os.walk(pkg_root):
        current_path = Path(current_path)

        # only process valid Python packages
        if "__init__.py" not in files:
            continue

        entries = generate_public_entries(current_path)
        replace_or_append_all(current_path / "__init__.py", entries)

# === Run ===
if __name__ == "__main__":
    process_package_recursively(ROOT_PACKAGE)