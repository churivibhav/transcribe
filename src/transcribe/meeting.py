from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re


@dataclass(slots=True)
class MeetingPaths:
    directory: Path
    raw_wav: Path
    recording_mp3: Path
    transcript_md: Path
    summary_md: Path
    metadata_json: Path
    chunks_dir: Path
    started_at: datetime
    title: str
    slug: str


def slugify_title(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-+", "-", lowered)
    return lowered.strip("-") or "meeting"


def create_meeting(output_dir: Path, title: str | None) -> MeetingPaths:
    started_at = datetime.now().astimezone()
    timestamp = started_at.strftime("%Y-%m-%d-%H%M")
    clean_title = title.strip() if title else ""

    if clean_title:
        slug = f"{slugify_title(clean_title)}-{timestamp}"
        final_title = clean_title
    else:
        slug = timestamp
        final_title = started_at.strftime("%Y-%m-%d %H:%M")

    directory = output_dir / slug
    chunks_dir = directory / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    return MeetingPaths(
        directory=directory,
        raw_wav=directory / "capture.wav",
        recording_mp3=directory / "recording.mp3",
        transcript_md=directory / "transcript.md",
        summary_md=directory / "summary.md",
        metadata_json=directory / "metadata.json",
        chunks_dir=chunks_dir,
        started_at=started_at,
        title=final_title,
        slug=slug,
    )


def write_metadata(paths: MeetingPaths, payload: dict) -> None:
    paths.metadata_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
