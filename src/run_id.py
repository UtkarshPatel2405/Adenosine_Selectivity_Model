import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional


def generate_run_id(prefix: str = "ADENO") -> str:
    """
    Generate a unique run ID.

    Format: {prefix}_{YYYYMMDD}_{HHMMSS}_{short_hash}
    Example: ADENO_20260622_143021_a3f7c2
    """
    now = time.time_ns()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now // 10**9))
    hash_input = f"{now}_{os.getpid()}_{os.urandom(4).hex()}"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:6]
    return f"{prefix}_{timestamp}_{short_hash}"


def run_id_filename(base_name: str, run_id: str, ext: str = ".json") -> str:
    """
    Generate a unique barcoded filename.

    Instead of 'evaluation.json' -> 'ADENO_20260622_143021_a3f7c2_evaluation.json'
    This makes every output file traceable to its generating run.

    Args:
        base_name: The logical name (e.g., "evaluation", "training_summary", "A1_report")
        run_id: The run identifier
        ext: File extension including dot (default: ".json")

    Returns:
        Barcoded filename string
    """
    return f"{run_id}_{base_name}{ext}"


def save_with_run_id(data, directory: Path, base_name: str, run_id: str, ext: str = ".json") -> Path:
    """
    Save data to a run-ID-barcoded file and create a 'latest' pointer.

    The actual file:       {directory}/{run_id}_{base_name}.json
    The 'latest' pointer:  {directory}/{base_name}.json  (a pointer JSON, not the data)

    Returns the path to the actual data file.
    """
    directory.mkdir(parents=True, exist_ok=True)

    actual_filename = run_id_filename(base_name, run_id, ext)
    actual_path = directory / actual_filename

    if ext == ".json":
        with open(actual_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    else:
        # Binary mode for images, etc.
        with open(actual_path, "wb") as f:
            f.write(data)

    # Write a 'latest' pointer JSON (works cross-platform, unlike symlinks)
    pointer = {
        "run_id": run_id,
        "actual_file": actual_filename,
        "full_path": str(actual_path.resolve()),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    pointer_path = directory / f"{base_name}{ext}"
    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump(pointer, f, indent=2)

    return actual_path


def read_latest(directory: Path, base_name: str, ext: str = ".json"):
    """
    Read the latest data file by following the 'latest' pointer.
    Returns (run_id, data) or (None, None) if not found.
    """
    pointer_path = directory / f"{base_name}{ext}"
    if not pointer_path.exists():
        return None, None

    try:
        with open(pointer_path) as f:
            pointer = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None, None

    actual_path = directory / pointer.get("actual_file", "")
    if not actual_path.exists():
        return None, None

    try:
        with open(actual_path) as f:
            data = json.load(f)
        return pointer.get("run_id"), data
    except (json.JSONDecodeError, FileNotFoundError):
        return None, None


def register_run(run_id: str, metadata: dict = None) -> Path:
    """
    Register a run in the run registry (outputs/run_registry.json).

    Appends an entry with run_id, timestamp, and optional metadata.
    Returns the path to the registry file.
    """
    from src.config import OUTPUTS_DIR

    registry_dir = OUTPUTS_DIR / "run_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / "run_registry.json"

    entry = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": metadata or {},
    }

    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = []

    registry.append(entry)

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    return registry_path


def get_run_history(directory: Path, base_name: str) -> list:
    """
    List all historical run files for a given base_name.
    Returns a list of dicts with {run_id, filename, mtime}.
    """
    pattern = f"*_{base_name}.*"
    files = []
    for fpath in directory.glob(pattern):
        # Extract run_id from filename like ADENO_20260622_143021_a3f7c2_evaluation.json
        parts = fpath.stem.split("_", 3)
        if len(parts) >= 4:
            run_id = "_".join(parts[:3])
            files.append({
                "run_id": run_id,
                "filename": fpath.name,
                "mtime": fpath.stat().st_mtime,
            })
    return sorted(files, key=lambda x: x["mtime"], reverse=True)
