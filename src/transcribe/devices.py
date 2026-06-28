from __future__ import annotations

from dataclasses import dataclass

from .system import run_command


@dataclass(slots=True)
class AudioSource:
    id: str
    name: str
    kind: str


def list_sources() -> list[AudioSource]:
    result = run_command(["pactl", "list", "sources", "short"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Unable to list PulseAudio/PipeWire sources.")

    sources: list[AudioSource] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        source_name = parts[1].strip()
        source_id = parts[0].strip()
        kind = classify_source(source_name)
        sources.append(AudioSource(id=source_id, name=source_name, kind=kind))

    return sources


def get_default_source_name() -> str:
    result = run_command(["pactl", "get-default-source"])
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def get_default_sink_monitor_name() -> str:
    result = run_command(["pactl", "get-default-sink"])
    if result.returncode != 0:
        return ""
    sink_name = result.stdout.strip()
    return f"{sink_name}.monitor" if sink_name else ""


def classify_source(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".monitor") or "monitor" in lowered:
        return "system"
    if "input" in lowered or "mic" in lowered:
        return "mic"
    return "other"
