#!/usr/bin/env python3
"""
file_structures.py

example usage in powershell:
    python host/tools/file_structures.py

example usage in powershell for full listing:
    python host/tools/file_structures.py --full

Generates a compact file tree view of the project and writes it to a text file.

Features:
  - Ignores some directories globally (e.g., .venv, .git, .vscode).
  - For specific "simplified" directories (e.g. ImageNet subset):
        data/datasets/imagenet1k_val_subset/ILSVRC2012_img_val_subset/
    it only descends into the FIRST 5 and LAST 5 subdirectories,
    and recursively prints their contents.

Default output:
    project_tree_YYYYMMDD.txt
"""

import argparse
from pathlib import Path
from datetime import datetime

# Directories to ignore entirely (by name)
IGNORE_DIR_NAMES = {".venv", ".git", ".vscode"}

# Directories to show in simplified form (paths relative to project root)
SIMPLIFIED_DIRS = [
    Path("data/datasets/imagenet1k_val_subset/ILSVRC2012_img_val_subset/")
]


def is_ignored_dir(path: Path) -> bool:
    """Return True if this directory should be globally ignored."""
    return path.is_dir() and path.name in IGNORE_DIR_NAMES


def is_simplified_dir(path: Path, root: Path) -> bool:
    """Return True if this directory matches one of the SIMPLIFIED_DIRS."""
    # Compare using paths relative to the project root for robustness
    rel = path.relative_to(root)
    for sdir in SIMPLIFIED_DIRS:
        if rel == sdir:
            return True
    return False


def list_directory_compact(path: Path, prefix: str, out_lines, root: Path):
    """
    Recursively list directories and files in a compact tree-like structure,
    respecting IGNORE_DIR_NAMES and SIMPLIFIED_DIRS.
    """
    if not path.exists():
        out_lines.append(f"{prefix}[Missing] {path}")
        return

    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

    for entry in entries:
        # Skip ignored directories
        if is_ignored_dir(entry):
            continue

        # Print the entry name
        out_lines.append(prefix + entry.name)

        # Handle directories
        if entry.is_dir():
            # If this directory is one of the simplified ones, use special logic
            if is_simplified_dir(entry, root):
                list_simplified_directory(entry, prefix + "    ", out_lines, root)
            else:
                list_directory_compact(entry, prefix + "    ", out_lines, root)


def list_simplified_directory(path: Path, prefix: str, out_lines, root: Path):
    """
    For a "simplified" directory (e.g., ImageNet subset), show only the first
    5 and last 5 subdirectories, but recurse fully into those 10 (or fewer).
    Files directly under 'path' are listed normally.
    """
    if not path.exists():
        out_lines.append(prefix + f"[Missing] {path}")
        return

    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    files = [e for e in entries if e.is_file()]
    dirs = [e for e in entries if e.is_dir() and not is_ignored_dir(e)]

    # List all files directly under this directory
    for f in files:
        out_lines.append(prefix + f.name)

    total_dirs = len(dirs)
    if total_dirs == 0:
        return

    if total_dirs <= 10:
        # If there are 10 or fewer dirs, just list them all normally
        for d in dirs:
            out_lines.append(prefix + d.name)
            list_directory_compact(d, prefix + "    ", out_lines, root)
    else:
        # First 5
        head = dirs[:5]
        # Last 5
        tail = dirs[-5:]

        for d in head:
            out_lines.append(prefix + d.name)
            list_directory_compact(d, prefix + "    ", out_lines, root)

        out_lines.append(prefix + "...")

        for d in tail:
            out_lines.append(prefix + d.name)
            list_directory_compact(d, prefix + "    ", out_lines, root)


def list_project_tree(root: Path, simplified: bool):
    """
    Generate the project directory tree as a list of text lines.
    If simplified=True, "simplified" dirs use the special partial-recursion logic.
    """
    out_lines = []

    out_lines.append("========== Project Tree ==========")
    out_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out_lines.append(f"Root: {root.resolve()}")
    out_lines.append("")

    # Top-level entries under root
    entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for item in entries:
        # Skip ignored dirs at top level as well
        if is_ignored_dir(item):
            continue

        out_lines.append(item.name)

        if item.is_dir():
            # If simplified mode is off, just recurse normally
            if not simplified:
                list_directory_compact(item, "    ", out_lines, root)
            else:
                # If this is a simplified dir, use special logic
                if is_simplified_dir(item, root):
                    list_simplified_directory(item, "    ", out_lines, root)
                else:
                    list_directory_compact(item, "    ", out_lines, root)

    out_lines.append("")
    out_lines.append("==================================")
    out_lines.append("")

    return out_lines


def main():
    parser = argparse.ArgumentParser(description="Project file structure viewer.")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory to analyze (default: project root)."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Disable simplified mode for special directories (e.g. ImageNet)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file name (overrides date-based default)."
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()

    if not root.exists():
        print(f"[ERROR] Root path not found: {root}")
        return

    # Auto-generate filename with today's date if not specified
    if args.out is None:
        date_str = datetime.now().strftime("%Y%m%d")
        out_path = Path(f"project_tree_{date_str}.txt")
    else:
        out_path = Path(args.out)

    simplified = not args.full
    lines = list_project_tree(root, simplified=simplified)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] File structure saved to: {out_path}")


if __name__ == "__main__":
    main()
