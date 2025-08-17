import os
import re
import ast
import argparse
from pathlib import Path

# === Customizable exclusion rules ===
EXCLUDE_ALL = {"Core", "SuportData"}
EXCLUDE_DIRS = {"__pycache__", "__main__", "_logging", "tests", "scripts"}

VERSION_PATTERN = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.M)
AUTHOR_PATTERN = re.compile(r"^__author__\s*=\s*['\"]([^'\"]+)['\"]", re.M)

summary_report = {"updated_modules": [], "skipped_modules": [], "updated_inits": [], "undocumented": {}}


def extract_metadata(init_path: Path, default_version="0.0.1", default_author="felavila") -> tuple[str, str]:
    if not init_path.exists():
        print(f"Warning: {init_path} not found. Using defaults.")
        return default_version, default_author

    content = init_path.read_text()
    version = VERSION_PATTERN.search(content)
    author = AUTHOR_PATTERN.search(content)
    return (
        version.group(1) if version else default_version,
        author.group(1) if author else default_author
    )


def ensure_metadata_lines(init_path: Path, version: str, author: str, dry_run: bool):
    content = init_path.read_text()
    lines = content.splitlines()

    modified = False

    if "__version__" in content:
        lines.insert(0, "")
        modified = True
      
    #if "__version__" not in content:
     #   lines.insert(0, f"__version__ = '{version}'")
      #  modified = True
    #else:
     #   lines = [re.sub(r"^__version__\s*=.*", f"__version__ = '{version}'", line) if "__version__" in line else line for line in lines]

    if "__author__" not in content:
        lines.insert(1, f"__author__ = '{author}'")
        modified = True
    if "__author__" in content:
        lines.insert(1, f"__author__ = '{author}'")
        modified = True

    if modified:
        updated = "\n".join(lines) + "\n"
        summary_report["updated_inits"].append(str(init_path))
        if not dry_run:
            init_path.write_text(updated)


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


def extract_public_names_and_docs(file_path: Path) -> tuple[list[str], list[str]]:
    public_names = []
    undocumented = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                public_names.append(node.name)
                if not ast.get_docstring(node):
                    undocumented.append(node.name)
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if not name.startswith("_"):
                    public_names.append(name)
    except Exception as e:
        print(f"Could not parse {file_path}: {e}")
    return sorted(public_names), sorted(undocumented)


def extract_all_entries(content: str) -> list[str]:
    match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.S)
    return re.findall(r'"(.*?)"', match.group(1)) if match else []


def write_all_to_module(file_path: Path, public_names: list[str], version: str, author: str, dry_run: bool, force: bool, check_docs: bool):
    content = file_path.read_text()
    lines = content.splitlines()
    existing_all = extract_all_entries(content)

    if not lines or not lines[0].strip().startswith('"""'):
        lines.insert(0, '"""This module handles ?."""')

    if "__version__" not in content:
        lines.insert(1, f"__version__ = '{version}'")
    if "__author__" not in content:
        lines.insert(2, f"__author__ = '{author}'")

    new_all = "# Auto-generated __all__\n__all__ = [\n" + "\n".join(f'    "{name}",' for name in public_names) + "\n]\n"
    full_content = "\n".join(lines)

    if "# Auto-generated __all__" in full_content or force:
        updated = re.sub(r"(?s)(# Auto-generated __all__\n)?__all__\s*=\s*\[.*?\]", new_all.strip(), full_content)
        summary_report["updated_modules"].append(str(file_path))
    elif "__all__" not in full_content:
        updated = full_content + "\n\n" + new_all
        summary_report["updated_modules"].append(str(file_path))
    elif sorted(existing_all) != public_names:
        print(f"Mismatch in {file_path} between __all__ and actual public names.")
        print("  Existing:", sorted(existing_all))
        print("  Expected:", public_names)
        summary_report["skipped_modules"].append(str(file_path))
        return
    else:
        summary_report["skipped_modules"].append(str(file_path))
        return

    if check_docs:
        _, undocumented = extract_public_names_and_docs(file_path)
        if undocumented:
            summary_report["undocumented"][str(file_path)] = undocumented

    if not dry_run:
        file_path.write_text(updated + "\n")


def process_module_files(pkg_path: Path, version: str, author: str, dry_run: bool, force: bool, check_docs: bool):
    for file in pkg_path.glob("*.py"):
        if file.name == "__init__.py":
            continue
        public_names, _ = extract_public_names_and_docs(file)
        if public_names:
            write_all_to_module(file, public_names, version, author, dry_run, force, check_docs)


def replace_or_append_all(init_path: Path, entries: list[str], dry_run: bool, force: bool):
    all_block = "# Auto-generated __all__\n__all__ = [\n" + "\n".join(f'    "{e}",' for e in entries) + "\n]\n"
    content = init_path.read_text()

    if "# Auto-generated __all__" in content or force:
        updated = re.sub(r"(?s)(# Auto-generated __all__\n)?__all__\s*=\s*\[.*?\]", all_block.strip(), content)
        summary_report["updated_inits"].append(str(init_path))
    elif "__all__" not in content:
        updated = content.strip() + "\n\n" + all_block
        summary_report["updated_inits"].append(str(init_path))
    else:
        return  # skip manual __all__

    if not dry_run:
        init_path.write_text(updated + "\n")


def process_package_recursively(pkg_root: Path, dry_run: bool, force: bool, check_docs: bool):
    root_init = pkg_root / "__init__.py"
    version, author = extract_metadata(root_init)

    for current_path, dirs, files in os.walk(pkg_root):
        current_path = Path(current_path)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        if "__init__.py" not in files:
            continue

        init_path = current_path / "__init__.py"
        rel_path = current_path.relative_to(pkg_root)
        module_name = rel_path.parts[0] if rel_path.parts else ""

        ensure_metadata_lines(init_path, version, author, dry_run)

        if module_name not in EXCLUDE_ALL:
            entries = generate_public_entries(current_path)
            replace_or_append_all(init_path, entries, dry_run, force)

        process_module_files(current_path, version, author, dry_run, force, check_docs)


def run_script(root_path: Path, dry_run: bool = False, force: bool = False, check_docs: bool = False):
    process_package_recursively(root_path, dry_run, force, check_docs)
    print(root_path)
    print("\nSummary:")
    print("Updated __init__.py files:")
    for path in summary_report["updated_inits"]:
        print("   •", path)
    print("Updated modules:")
    for path in summary_report["updated_modules"]:
        print("   •", path)
    print("Skipped modules (manual or mismatched __all__):")
    for path in summary_report["skipped_modules"]:
        print("   •", path)
    if check_docs:
        print("Undocumented functions/classes:")
        for mod, items in summary_report["undocumented"].items():
            print(f"   • {mod}: {', '.join(items)}")


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-update __all__, __version__, __author__ in Python modules.")
    parser.add_argument("--path", type=str, required=True, help="Path to the root of your package")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument("--force", action="store_true", help="Force overwrite even if __all__ exists or is manual")
    parser.add_argument("--check-docs", action="store_true", help="Report public functions/classes missing docstrings")

    args = parser.parse_args()
    run_script(Path(args.path), dry_run=args.dry_run, force=args.force, check_docs=args.check_docs)

"""
USAGE EXAMPLES
--------------

# 1. Basic usage:
python all_editor.py --path sheap/

# 2. Simulate changes (dry run):
python all_editor.py --path sheap/ --dry-run

# 3. Force overwrite existing __all__ blocks:
python all_editor.py --path sheap/ --force

# 4. Also check for missing docstrings:
python all_editor.py --path sheap/ --check-docs

# 5. All features combined:
python all_editor.py --path sheap/ --force --check-docs --dry-run
"""
