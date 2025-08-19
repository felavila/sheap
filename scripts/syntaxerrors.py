#!/usr/bin/env python3
"""
Check all .py file (catch SyntaxErrors).
"""

import ast
from pathlib import Path
import sys

def check_syntax(root: Path):
    errors = []
    for py_file in root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
            ast.parse(source, filename=str(py_file))
        except SyntaxError as e:
            errors.append((py_file, e.lineno, e.msg, e.text))
    return errors


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    errs = check_syntax(root)
    if not errs:
        print("✅ All Python files parsed without SyntaxError")
    else:
        for path, lineno, msg, text in errs:
            print(f"❌ {path}:{lineno} — {msg}")
            if text:
                print(f"    {text.strip()}")
        sys.exit(1)
