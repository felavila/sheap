import os
from pathlib import Path
import re
import ast

# === Customizable ===
EXCLUDE_ALL = {"Core", "SuportData"}  # skip __init__.py __all__ generation, but still add __version__
EXCLUDE_DIRS = {"__pycache__", "__main__", "_logging", "tests", "scripts"}
ROOT_PACKAGE = Path(__file__).resolve().parent.parent / "sheap"
VERSION_PATTERN = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.M)

# === Extract __version__ from root __init__.py ===
def extract_version(init_path: Path, default: str = "0.0.1") -> str:
    if not init_path.exists():
        print(f"Warning: {init_path} not found. Using default version: {default}")
        return default
    content = init_path.read_text()
    match = VERSION_PATTERN.search(content)
    if match:
        return match.group(1)
    print(f"Warning: No __version__ found in {init_path}. Using default version: {default}")
    return default

# === Insert or update __version__ = ... ===
def ensure_version_line(init_path: Path, version: str):
    content = init_path.read_text()
    version_line = f"__version__ = '{version}'"

    if "__version__" in content:
        updated = re.sub(
            r"^__version__\s*=\s*['\"][^'\"]+['\"]",
            version_line,
            content,
            flags=re.M
        )
    else:
        updated = version_line + "\n" + content

    init_path.write_text(updated)

# === List public .py modules and subpackages ===
def generate_public_entries(pkg_path: Path) -> list[str]:
    entries = []
    for item in sorted(pkg_path.iterdir()):
        name = item.stem if item.is_file() else item.name
        if name.startswith("_") or name in EXCLUDE_ALL or name in EXCLUDE_DIRS:
            continue
        if item.is_file() and item.suffix == ".py":
            entries.append(name)
        elif item.is_dir() and (item / "__init__.py").exists():
            entries.append(name)
    return sorted(entries)

# === Replace or add __all__ block in __init__.py ===
def replace_or_append_all(init_path: Path, entries: list[str]):
    new_all_block = "# Auto-generated __all__\n__all__ = [\n" + "\n".join(f'    "{e}",' for e in entries) + "\n]\n"
    content = init_path.read_text()

    if "# Auto-generated __all__" in content:
        updated = re.sub(r"(?s)# Auto-generated __all__\n__all__\s*=\s*\[.*?\]", new_all_block.strip(), content)
    elif "__all__" in content:
        # leave user-defined __all__ untouched
        updated = content
    else:
        updated = content.strip() + "\n\n" + new_all_block

    init_path.write_text(updated + "\n")

# === Extract public names from a single .py file ===
def extract_public_names_from_file(file_path: Path) -> list[str]:
    public_names = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    public_names.append(node.name)
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if not name.startswith("_"):
                    public_names.append(name)
    except Exception as e:
        print(f"Could not parse {file_path}: {e}")
    return sorted(public_names)

# === Write or update __all__ block in a module ===
def write_all_to_module(file_path: Path, public_names: list[str]):
    new_all = "# Auto-generated __all__\n__all__ = [\n" + "\n".join(f'    "{name}",' for name in public_names) + "\n]\n"
    content = file_path.read_text()

    if "# Auto-generated __all__" in content:
        updated = re.sub(r"(?s)# Auto-generated __all__\n__all__\s*=\s*\[.*?\]", new_all.strip(), content)
    elif "__all__" in content:
        # leave manual __all__ untouched
        updated = content
    else:
        updated = new_all + "\n" + content

    file_path.write_text(updated)

# === Update all .py files inside a folder (except __init__.py) ===
def process_module_files(pkg_path: Path):
    for file in pkg_path.glob("*.py"):
        if file.name == "__init__.py":
            continue
        public_names = extract_public_names_from_file(file)
        if public_names:
            write_all_to_module(file, public_names)

# === Main recursive function ===
def process_package_recursively(pkg_root: Path):
    root_init = pkg_root / "__init__.py"
    version = extract_version(root_init)

    for current_path, dirs, files in os.walk(pkg_root):
        current_path = Path(current_path)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        if "__init__.py" not in files:
            continue

        init_path = current_path / "__init__.py"
        rel_path = current_path.relative_to(pkg_root)
        module_name = rel_path.parts[0] if rel_path.parts else ""

        if init_path != root_init:
            ensure_version_line(init_path, version)

        if module_name not in EXCLUDE_ALL:
            entries = generate_public_entries(current_path)
            replace_or_append_all(init_path, entries)

        process_module_files(current_path)

# === Run ===
if __name__ == "__main__":
    print(f"Processing: {ROOT_PACKAGE}")
    process_package_recursively(ROOT_PACKAGE)
    print("Done.")
