from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    text: str


def seconds_to_clock(total_seconds: float) -> str:
    rounded = max(0, int(total_seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def render_transcript(
    title: str,
    started_at: datetime,
    duration_seconds: float | None,
    model_name: str,
    device: str,
    compute_type: str,
    segments: list[TranscriptSegment],
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Model: faster-whisper {model_name}",
        f"Device: {device}",
        f"Compute type: {compute_type}",
    ]

    if duration_seconds is not None:
        lines.append(f"Duration: {seconds_to_clock(duration_seconds)}")

    lines.extend(["", "## Transcript", ""])

    if not segments:
        lines.append("No transcript generated.")
    else:
        for segment in segments:
            lines.append(f"[{seconds_to_clock(segment.start_seconds)}] {segment.text}")

    lines.append("")
    return "\n".join(lines)
