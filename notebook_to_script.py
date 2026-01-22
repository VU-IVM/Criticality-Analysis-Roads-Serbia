import json
from pathlib import Path


def notebook_to_script(ipynb_path, py_path=None):
    ipynb_path = Path(ipynb_path)

    if py_path is None:
        py_path = ipynb_path.with_suffix(".py")

    with ipynb_path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    with py_path.open("w", encoding="utf-8") as f:
        f.write(f"# Converted from {ipynb_path.name}\n\n")

        cell_counter = 1

        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                f.write(f"\n# ===== Cell {cell_counter} =====\n")
                for line in cell.get("source", []):
                    f.write(line)
                f.write("\n")
                cell_counter += 1

    print(f"Written: {py_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: py notebook_to_script.py <notebook.ipynb>")
        sys.exit(1)

    notebook_to_script(sys.argv[1])
