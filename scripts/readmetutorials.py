from pathlib import Path

root = Path("tutorials")
lines = ["# Tutorials — Open in Colab\n"]
for nb in sorted(root.rglob("*.ipynb")):
    rel = nb.as_posix()
    title = rel.removeprefix("tutorials/")
    url = f"https://colab.research.google.com/github/felavila/sheap/blob/main/{rel}"
    lines += [f"### {title}\n", f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})\n"]
Path("tutorials/README.md").write_text("\n".join(lines))
print("Wrote tutorials/README.md")