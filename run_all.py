import subprocess
import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"

scripts = sorted(src_path.glob("*.py"))

for script in scripts:
    print(f"Running: {script.name}")
    result = subprocess.run([sys.executable, str(script)], check=True)
    print(f"Finished: {script.name}\n")