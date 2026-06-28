from __future__ import annotations

import shutil
import subprocess


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)
