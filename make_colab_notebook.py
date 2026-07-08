"""Generate a Google Colab notebook that loads dataset from Google Drive."""
# python make_colab_notebook.py

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "SLAL_chatbot.ipynb"

PYTHON_FILES = ["build_index.py", "chatbot.py", "app.py"]
DRIVE_DATASET_PATH = "/content/drive/MyDrive/MSc/NLP/Chatbot/dataset"
PROJECT_DIR = "/content/SLAL_chatbot"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def code_cell(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def markdown_cell(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def build_code_files_cell() -> str:
    files = {name: read_text(ROOT / name) for name in PYTHON_FILES}
    cell = """import os
from pathlib import Path

PROJECT_DIR = Path("__PROJECT_DIR__")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_DIR)

CODE_FILES = __CODE_FILES__

for filename, content in CODE_FILES.items():
    (PROJECT_DIR / filename).write_text(content, encoding="utf-8")

print(f"Project directory: {PROJECT_DIR}")
print("Created:", sorted(CODE_FILES.keys()))
"""
    return cell.replace("__PROJECT_DIR__", PROJECT_DIR).replace(
        "__CODE_FILES__", repr(files)
    )


def main() -> None:
    cells = [
        markdown_cell(
            """# Sri Lankan Airlines Policy Assistant

## What this notebook recreates automatically
- `build_index.py`, `chatbot.py`, `app.py`
- `vector_store/` (FAISS index built from the dataset)
"""
        ),
        code_cell(
            """from google.colab import drive
drive.mount("/content/drive")"""
        ),
        code_cell(
            f"""import os
import shutil
from pathlib import Path

DRIVE_DATASET_PATH = Path("{DRIVE_DATASET_PATH}")
PROJECT_DIR = Path("{PROJECT_DIR}")
LOCAL_DATASET_PATH = PROJECT_DIR / "dataset"

if not DRIVE_DATASET_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found at: " + str(DRIVE_DATASET_PATH) + "\\n"
        "Upload your dataset folder to: My Drive > MSc > NLP > Chatbot > dataset"
    )

PROJECT_DIR.mkdir(parents=True, exist_ok=True)

if LOCAL_DATASET_PATH.exists():
    shutil.rmtree(LOCAL_DATASET_PATH)

shutil.copytree(DRIVE_DATASET_PATH, LOCAL_DATASET_PATH)

txt_files = sorted(LOCAL_DATASET_PATH.glob("*.txt"))
print(f"Copied {{len(txt_files)}} dataset files to {{LOCAL_DATASET_PATH}}")
for path in txt_files[:5]:
    print(" -", path.name)
if len(txt_files) > 5:
    print(f" ... and {{len(txt_files) - 5}} more")
"""
        ),
        code_cell(build_code_files_cell()),
        code_cell(
            "!pip install -q langchain langchain-community langchain-huggingface "
            "faiss-cpu sentence-transformers transformers torch gradio"
        ),
        code_cell("!python build_index.py"),
        code_cell(
            f"""import sys
from pathlib import Path

sys.path.insert(0, "{PROJECT_DIR}")

from app import demo

demo.launch(share=True)"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
            "colab": {
                "provenance": [],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUTPUT.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    json.loads(OUTPUT.read_text(encoding="utf-8"))

    print(f"Wrote {OUTPUT}")
    print(f"Embedded Python files: {', '.join(PYTHON_FILES)}")
    print(f"Drive dataset path: {DRIVE_DATASET_PATH}")


if __name__ == "__main__":
    main()
