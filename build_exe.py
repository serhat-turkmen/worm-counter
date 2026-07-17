"""
Build a standalone desktop executable of the Worm Counter (Flask version).

Usage:
    pip install pyinstaller
    python build_exe.py

Output:
    dist/WormCounter/WormCounter   (macOS/Linux)
    dist/WormCounter/WormCounter.exe  (Windows)

Users can then just double-click the executable — no Python required.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.join(ROOT, "worm_app", "app.py")
TEMPLATES = os.path.join(ROOT, "worm_app", "templates")
OUT_NAME = "WormCounter"

# separator is ; on Windows, : on macOS/Linux
SEP = ";" if sys.platform == "win32" else ":"

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onedir",                          # folder bundle (more reliable than --onefile for Flask)
    "--name", OUT_NAME,
    "--console",                         # keep console for server log output
    # Include the templates directory
    f"--add-data={TEMPLATES}{SEP}templates",
    # Hidden imports that PyInstaller may miss
    "--hidden-import=flask",
    "--hidden-import=flask_cors",
    "--hidden-import=cv2",
    "--hidden-import=numpy",
    "--hidden-import=tkinter",
    "--hidden-import=tkinter.filedialog",
    # Exclude large unused packages to keep bundle small
    "--exclude-module=matplotlib",
    "--exclude-module=scipy",
    "--exclude-module=pandas",
    "--exclude-module=streamlit",
    ENTRY,
]

print("Building Worm Counter executable...")
print(f"Entry: {ENTRY}")
print(f"Templates: {TEMPLATES}")
print()

result = subprocess.run(cmd, cwd=ROOT)
if result.returncode != 0:
    print("\n❌ Build failed. Check output above.")
    sys.exit(1)

print(f"""
✅ Build complete!

Executable is in:  dist/{OUT_NAME}/

To distribute:
  Zip the entire dist/{OUT_NAME}/ folder and send to colleagues.
  They can run {OUT_NAME} (or {OUT_NAME}.exe on Windows) directly.

The app will open at http://localhost:8080 in their browser.
""")
