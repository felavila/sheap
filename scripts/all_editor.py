import os
import re
import ast
import argparse
from pathlib import Path

# === Customizable exclusion rules ===
EXCLUDE_ALL = {"Core", "SuportData"}
EXCLUDE_DIRS = {"__pycache__", "__main__", "_logging", "tests", "scripts"}

AUTHOR_PATTERN = re.compile(r"^__author__\s*=\s*['\"]([^'\"]+)['\"]", re.M)

summary_report = {"updated_modules": [], "skipped_modules": [], "updated_inits": [], "undocumented": {}}


def extract_metadata(init_path: Path, default_author="Unknown") -> str:
    """Extract author from a file, default if missing."""
    if not init_path.exists():
        print(f"Warning: {init_path} not found. Using defaults.")
        return default_author

    content = init_path.read_text()
    author = AUTHOR_PATTERN.search(content)
    return author.group(1) if author else default_author


def ensure_metadata_lines(init_path: Path, author: str, dry_run: bool):
    """Clean __version__, set/replace __author__ in __init__.py files."""
    content = init_path.read_text()
    lines = content.splitlines()
    modified = False

    # Remove ALL __version__ lines
    new_lines = [line for line in lines if not line.strip().startswith("__version__")]
    if len(new_lines) != len(lines):
        modified = True
    lines = new_lines

    # Replace or add __author__
    author_line = f"__author__ = '{author}'"
    found_author = False
    for i, line in enumerate(lines):
        if line.strip().startswith("__author__"):
            if line.strip() != author_line:
                lines[i] = author_line
                modified = True
            found_author = True
            break
    if not found_author:
        lines.insert(0, author_line)
        modified = True

    if modified:
        updated = "\n".join(lines) + "\n"
        summary_report["updated_inits"].append(str(init_path))
        if not dry_run:
            init_path.write_text(updated)


def generate_public_entries(pkg_path: Path) -> list[str]:
    """Find submodules/subpackages to expose in __all__."""
    entries = []
    for item in sorted(pkg_path.iterdir()):
        name = item.stem if item.is_file() else item.name
        if name == "__init__":
            continue
        if name in EXCLUDE_ALL or name in EXCLUDE_DIRS:
            continue
        if item.is_file() and item.suffix == ".py":
            entries.append(name)
        elif item.is_dir() and (item / "__init__.py").exists():
            entries.append(name)
    return sorted(entries)


def is_public(name: str) -> bool:
    """Return True if the symbol should be exported in __all__."""
    # Exclude private (_foo) and dunder (__foo__)
    if name.startswith("_"):
        return False
    if name.startswith("__") and name.endswith("__"):
        return False
    return True


def extract_public_names_and_docs(file_path: Path) -> tuple[list[str], list[str]]:
    """
    Extract top-level public function/class/variable names from a Python file.

    Returns
    -------
    public_names : list of str
        Names that should go into __all__.
    undocumented : list of str
        Public names missing a docstring (for reporting).
    """
    public_names = []
    undocumented = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        for node in tree.body:
            # Functions and classes
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not is_public(node.name):
                    continue
                public_names.append(node.name)
                if not ast.get_docstring(node):
                    undocumented.append(node.name)

            # Module-level variables
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if not is_public(name):
                    continue
                public_names.append(name)

    except Exception as e:
        print(f"Could not parse {file_path}: {e}")

    return sorted(public_names), sorted(undocumented)





def extract_all_entries(content: str) -> list[str]:
    match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.S)
    return re.findall(r'"(.*?)"', match.group(1)) if match else []


def write_all_to_module(file_path: Path, public_names: list[str], author: str,
                        dry_run: bool, force: bool, check_docs: bool):
    """Remove __version__, replace __author__, ensure __all__."""
    content = file_path.read_text()
    lines = content.splitlines()

    # Remove __version__
    lines = [line for line in lines if not line.strip().startswith("__version__")]

    # Replace or add __author__
    author_line = f"__author__ = '{author}'"
    found_author = False
    for i, line in enumerate(lines):
        if line.strip().startswith("__author__"):
            if line.strip() != author_line:
                lines[i] = author_line
            found_author = True
            break
    if not found_author:
        lines.insert(0, author_line)

    # Ensure module docstring exists
    if not lines or not lines[0].strip().startswith('"""'):
        lines.insert(0, '"""This module handles ?."""')

    # Build new __all__
    new_all = "__all__ = [\n" + \
              "\n".join(f'    "{name}",' for name in sorted(set(public_names))) + "\n]\n"
    full_content = "\n".join(lines)
    existing_all = extract_all_entries(full_content)

    if "__all__" not in full_content:
        updated = full_content + "\n\n" + new_all
        summary_report["updated_modules"].append(str(file_path))
    elif sorted(existing_all) != sorted(public_names) or force:
        updated = re.sub(r"(?s)__all__\s*=\s*\[.*?\]", new_all.strip(), full_content)
        summary_report["updated_modules"].append(str(file_path))
    else:
        summary_report["skipped_modules"].append(str(file_path))
        return

    if check_docs:
        _, undocumented = extract_public_names_and_docs(file_path)
        if undocumented:
            summary_report["undocumented"][str(file_path)] = undocumented

    if not dry_run:
        file_path.write_text(updated + "\n")


def process_module_files(pkg_path: Path, author: str, dry_run: bool, force: bool, check_docs: bool):
    for file in pkg_path.glob("*.py"):
        if file.name == "__init__.py":
            continue
        public_names, _ = extract_public_names_and_docs(file)
        if public_names:
            write_all_to_module(file, public_names, author, dry_run, force, check_docs)


def replace_or_append_all(init_path: Path, entries: list[str], dry_run: bool, force: bool):
    all_block = "__all__ = [\n" + \
                "\n".join(f'    "{e}",' for e in sorted(set(entries))) + "\n]\n"
    content = init_path.read_text()
    existing_all = extract_all_entries(content)

    if "__all__" not in content:
        updated = content.strip() + "\n\n" + all_block
        summary_report["updated_inits"].append(str(init_path))
    elif sorted(existing_all) != sorted(entries) or force:
        updated = re.sub(r"(?s)__all__\s*=\s*\[.*?\]", all_block.strip(), content)
        summary_report["updated_inits"].append(str(init_path))
    else:
        return  # skip unchanged

    if not dry_run:
        init_path.write_text(updated + "\n")


def process_package_recursively(pkg_root: Path, dry_run: bool, force: bool, check_docs: bool, author: str):
    root_init = pkg_root / "__init__.py"
    author = extract_metadata(root_init, author)

    for current_path, dirs, files in os.walk(pkg_root):
        current_path = Path(current_path)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        if "__init__.py" not in files:
            continue

        init_path = current_path / "__init__.py"
        rel_path = current_path.relative_to(pkg_root)
        module_name = rel_path.parts[0] if rel_path.parts else ""

        ensure_metadata_lines(init_path, author, dry_run)

        if module_name not in EXCLUDE_ALL:
            entries = generate_public_entries(current_path)
            replace_or_append_all(init_path, entries, dry_run, force)

        process_module_files(current_path, author, dry_run, force, check_docs)


def run_script(root_path: Path, dry_run: bool = False, force: bool = False, check_docs: bool = False, author: str = "Unknown"):
    process_package_recursively(root_path, dry_run, force, check_docs, author)

    print("\nSummary:")
    print("Updated __init__.py files:")
    for path in summary_report["updated_inits"]:
        print("   •", path)
    print("Updated modules:")
    for path in summary_report["updated_modules"]:
        print("   •", path)
    print("Skipped modules (manual or unchanged __all__):")
    for path in summary_report["skipped_modules"]:
        print("   •", path)
    if check_docs:
        print("Undocumented functions/classes:")
        for mod, items in summary_report["undocumented"].items():
            print(f"   • {mod}: {', '.join(items)}")


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-update __all__, remove __version__, set __author__ in Python modules.")
    parser.add_argument("--path", type=str, required=True, help="Path to the root of your package")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument("--force", action="store_true", help="Force overwrite even if __all__ exists or is manual")
    parser.add_argument("--check-docs", action="store_true", help="Report public functions/classes missing docstrings")
    parser.add_argument("--author", type=str, default="Unknown", help="Set this as __author__ in all files")

    args = parser.parse_args()
    run_script(Path(args.path), dry_run=args.dry_run, force=args.force, check_docs=args.check_docs, author=args.author)
