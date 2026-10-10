
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_command(*args):
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_run_directory():
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_id = f"{timestamp}-{uuid4().hex[:8]}"

    output = Path("results") / run_id
    output.mkdir(parents=True, exist_ok=False)

    return output, run_id


def create_provenance(metadata, run_id):
    commit = git_command("rev-parse", "HEAD")
    status = git_command(
        "status", "--porcelain", "--untracked-files=all"
    )

    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in [
                    "scikit-learn",
                    "numpy",
                    "scipy",
                    "pandas",
                    "matplotlib",
                    "typer",
                ]
            },
        },
        "source": {
            "git_commit": commit,
            "git_dirty": (
                bool(status) if status is not None else None
            ),
            "git_status": status,
        },
        **metadata,
    }


def save_json(path, data):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)
