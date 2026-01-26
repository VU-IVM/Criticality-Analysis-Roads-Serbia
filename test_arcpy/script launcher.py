import arcpy
import subprocess
import os

external_python = (
    r"C:\Users\yma794\AppData\Local\miniforge3\envs\serbia\python.exe"
)

script_path = (
    r"C:\Users\yma794\Documents\Serbia\Analysis - Copy\1b_NetworkPreparation.py"
)

# --- clean environment ---
clean_env = os.environ.copy()

for var in (
    "PYTHONPATH",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_SHLVL",
):
    clean_env.pop(var, None)

# 🔑 FORCE matplotlib backend
clean_env["MPLBACKEND"] = "Agg"

arcpy.AddMessage("Running external Python (venv, isolated)...")

result = subprocess.run(
    [external_python, script_path],
    capture_output=True,
    text=True,
    env=clean_env
)

if result.returncode != 0:
    arcpy.AddError(result.stderr)
    raise RuntimeError("External script failed")

arcpy.AddMessage(result.stdout)
