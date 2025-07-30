import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_API_PATH = PROJECT_ROOT / "docs" / "source" / "api"

def clean_rst_file(filepath: str):
    with open(filepath, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    inside_autosummary = False
    skip_next_block = False
    inside_automodule_block = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # --- Detect automodule/autoclass start ---
        if stripped.startswith(".. auto") and "module" in stripped:
            inside_automodule_block = True
            cleaned_lines.append(line)
            continue

        # --- Fix misformatted automodule/autoclass options ---
        if inside_automodule_block:
            if stripped == "":
                inside_automodule_block = False
            elif stripped in {"members", "undoc-members", "show-inheritance"}:
                # Correct indentation and add colons
                line = f"   :{stripped}:\n"

        # --- Fix autosummary block ---
        if ".. autosummary::" in line:
            inside_autosummary = True
            skip_next_block = ":recursive:" in line
            cleaned_lines.append(line)
            continue

        if inside_autosummary:
            if stripped.startswith(":") or stripped == "":
                if ":recursive:" in stripped:
                    continue
                if skip_next_block:
                    continue
            elif not line.startswith(" "):
                inside_autosummary = False
                skip_next_block = False

        # --- Remove duplicate automodule block if submodules are present ---
        if ".. automodule::" in line:
            if any("Submodules" in l for l in lines[i:i + 10]):
                continue

        # --- Remove headings for submodules/subpackages ---
        if stripped in {"Submodules", "Subpackages"}:
            continue
        if stripped.startswith("~~~~~~~~~"):  # underline after heading
            continue

        cleaned_lines.append(line)

    with open(filepath, "w") as f:
        f.writelines(cleaned_lines)


def clean_all_rst():
    for file in os.listdir(DOCS_API_PATH):
        if file.endswith(".rst") and file.startswith("sheap"):
            print(f"Cleaning {file}")
            clean_rst_file(DOCS_API_PATH / file)

if __name__ == "__main__":
    clean_all_rst()
