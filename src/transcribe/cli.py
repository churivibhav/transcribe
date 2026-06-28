from __future__ import annotations

from pathlib import Path
import select
import shutil
import sys
import termios
import threading
import time
import tty
from contextlib import contextmanager

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import Settings
from .devices import get_default_sink_monitor_name, get_default_source_name, list_sources
from .markdown import TranscriptSegment, render_transcript, seconds_to_clock
from .meeting import create_meeting, write_metadata
from .notes import generate_summary
from .recorder import analyze_audio_volume, convert_wav_to_mp3, record_audio_sample, start_recording
from .system import command_exists
from .transcriber import ChunkTranscriber

app = typer.Typer(help="Linux meeting recorder and live transcription tool.")
console = Console()


class LiveSession:
    def __init__(self, settings: Settings, title: str | None, with_summary: bool) -> None:
        self.settings = settings
        self.paths = create_meeting(settings.output_dir(), title)
        self.with_summary = with_summary
        self.segments: list[TranscriptSegment] = []
        self.transcriber = ChunkTranscriber(settings)
        self._stop_event = threading.Event()
        self._error: str | None = None
        self._lock = threading.Lock()
        self.status = "Recording"

    def append_segments(self, new_segments: list[TranscriptSegment]) -> None:
        with self._lock:
            self.segments.extend(new_segments)
            self.segments.sort(key=lambda segment: segment.start_seconds)
        self.write_transcript(None)

    def write_transcript(self, duration_seconds: float | None) -> None:
        with self._lock:
            segments = list(self.segments)
        content = render_transcript(
            title=self.paths.title,
            started_at=self.paths.started_at,
            duration_seconds=duration_seconds,
            model_name=self.settings.whisper_model,
            device=self.settings.whisper_device,
            compute_type=self.settings.whisper_compute_type,
            segments=segments,
        )
        self.paths.transcript_md.write_text(content, encoding="utf-8")

    def render(self, elapsed_seconds: float) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_row(f"Title: {self.paths.title}")
        grid.add_row(f"Elapsed: {seconds_to_clock(elapsed_seconds)}")
        grid.add_row(f"Status: {self.status}")
        grid.add_row(f"Model: {self.settings.whisper_model} / {self.settings.whisper_device} / {self.settings.whisper_compute_type}")
        grid.add_row(f"Output: {self.paths.directory}")
        grid.add_row("Stop: press q to stop and save. Ctrl+C is emergency abort.")

        transcript = Text()
        with self._lock:
            recent_segments = self.segments[-20:]
        if recent_segments:
            for segment in recent_segments:
                transcript.append(f"[{seconds_to_clock(segment.start_seconds)}] ", style="cyan")
                transcript.append(segment.text)
                transcript.append("\n")
        else:
            transcript.append("Waiting for the first completed audio chunk...", style="dim")

        layout = Table.grid(expand=True)
        layout.add_column(ratio=1)
        layout.add_row(Panel(grid, title="Meeting", border_style="green"))
        layout.add_row(Panel(transcript, title="Live Transcript", border_style="blue"))
        return Panel(layout, border_style="white")

    def process_chunk_files(self) -> None:
        seen: set[Path] = set()
        while not self._stop_event.is_set():
            try:
                chunk_files = sorted(self.paths.chunks_dir.glob("chunk_*.wav"))
                for index, chunk_file in enumerate(chunk_files):
                    if chunk_file in seen:
                        continue
                    if not chunk_file.exists() or chunk_file.stat().st_size == 0:
                        continue
                    # During recording, skip the newest chunk because ffmpeg may still be writing it.
                    if index == len(chunk_files) - 1:
                        continue

                    chunk_index = int(chunk_file.stem.split("_")[-1])
                    offset = chunk_index * self.settings.chunk_seconds
                    new_segments = self.transcriber.transcribe_file(chunk_file, offset_seconds=offset)
                    seen.add(chunk_file)
                    if new_segments:
                        self.append_segments(new_segments)
            except Exception as exc:  # pragma: no cover - defensive runtime path
                self._error = str(exc)
                self._stop_event.set()
                break

            time.sleep(1)

        try:
            for chunk_file in sorted(self.paths.chunks_dir.glob("chunk_*.wav")):
                if chunk_file in seen or chunk_file.stat().st_size == 0:
                    continue
                chunk_index = int(chunk_file.stem.split("_")[-1])
                offset = chunk_index * self.settings.chunk_seconds
                new_segments = self.transcriber.transcribe_file(chunk_file, offset_seconds=offset)
                seen.add(chunk_file)
                if new_segments:
                    self.append_segments(new_segments)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self._error = str(exc)


def print_sources() -> None:
    sources = list_sources()
    if not sources:
        console.print("No audio sources found.")
        return

    table = Table(title="Audio Sources")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Name")
    for source in sources:
        table.add_row(source.kind, source.id, source.name)
    console.print(table)

    mic_candidates = [source.name for source in sources if source.kind == "mic"]
    system_candidates = [source.name for source in sources if source.kind == "system"]
    default_mic = get_default_source_name()
    default_system = get_default_sink_monitor_name()
    suggested_mic = default_mic if default_mic in mic_candidates else (mic_candidates[0] if mic_candidates else "")
    suggested_system = default_system if default_system in system_candidates else (system_candidates[0] if system_candidates else "")

    if default_mic:
        console.print(f"\nDefault mic source: {default_mic}")
    if default_system:
        console.print(f"Default system monitor: {default_system}")
    console.print("\nSuggested .env values:")
    console.print(f"AUDIO_MIC_SOURCE={suggested_mic}")
    console.print(f"AUDIO_SYSTEM_SOURCE={suggested_system}")


@contextmanager
def raw_terminal_input():
    if not sys.stdin.isatty():
        yield
        return

    original_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    return sys.stdin.read(1)


@app.command()
def doctor() -> None:
    """Check runtime dependencies and config.
    """

    settings = Settings()
    rows = [
        ("ffmpeg", command_exists("ffmpeg")),
        ("pactl", command_exists("pactl")),
    ]

    table = Table(title="Doctor")
    table.add_column("Check")
    table.add_column("Status")
    for name, ok in rows:
        table.add_row(name, "ok" if ok else "missing")
    console.print(table)

    console.print(f"Output directory: {settings.output_dir()}")
    if not settings.audio_mic_source or not settings.audio_system_source:
        console.print("Audio sources are not fully configured in .env")
    if settings.summary_enabled() and not settings.openai_api_key:
        console.print("SUMMARY_PROVIDER is enabled but OPENAI_API_KEY is missing")


@app.command()
def devices() -> None:
    """List available audio sources.
    """

    if not command_exists("pactl"):
        raise typer.BadParameter("pactl is not installed. Install pulseaudio-utils first.")
    print_sources()


@app.command("test-audio")
def test_audio(
    source: str = typer.Option(default="both", help="Audio source to test: mic, system, or both."),
    seconds: int = typer.Option(default=8, min=1, max=60, help="Seconds to record."),
) -> None:
    """Record a short sample to verify configured audio sources.
    """

    settings = Settings()
    source = source.strip().lower()
    if source not in {"mic", "system", "both"}:
        raise typer.BadParameter("source must be one of: mic, system, both")
    if not command_exists("ffmpeg"):
        raise typer.BadParameter("ffmpeg is not installed. Install it with: sudo apt install ffmpeg")
    if not settings.audio_mic_source or not settings.audio_system_source:
        raise typer.BadParameter("Set AUDIO_MIC_SOURCE and AUDIO_SYSTEM_SOURCE in .env. Run `transcribe devices` first.")

    settings.output_dir().mkdir(parents=True, exist_ok=True)
    meeting = create_meeting(settings.output_dir(), f"audio-test-{source}")
    sample_wav = meeting.directory / f"{source}.wav"
    sample_mp3 = meeting.directory / f"{source}.mp3"

    console.print(f"Recording {seconds}s {source} sample...")
    result = record_audio_sample(
        output_wav=sample_wav,
        mic_source=settings.audio_mic_source,
        system_source=settings.audio_system_source,
        seconds=seconds,
        source=source,
    )
    if result.returncode != 0:
        console.print(result.stderr.strip() or "Unable to record audio sample")
        raise typer.Exit(code=1)

    conversion = convert_wav_to_mp3(sample_wav, sample_mp3)
    if conversion.returncode != 0:
        console.print(conversion.stderr.strip() or "Unable to convert sample to MP3")
        raise typer.Exit(code=1)

    analysis = analyze_audio_volume(sample_wav)
    volume_lines = [line.strip() for line in analysis.stderr.splitlines() if "mean_volume" in line or "max_volume" in line]

    console.print(f"Saved sample: {sample_mp3}")
    if volume_lines:
        console.print("Volume diagnostics:")
        for line in volume_lines:
            console.print(line)
    else:
        console.print("No volume diagnostics were produced.")

    console.print("Play the sample to confirm it contains the expected audio.")


@app.command("transcribe-file")
def transcribe_file(
    audio_file: Path = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    title: str | None = typer.Option(default=None, help="Optional meeting title."),
    summary: bool = typer.Option(default=False, help="Generate summary with configured provider."),
    language: str | None = typer.Option(default=None, help="Language hint, for example auto, en, mr, hi, fr, nl."),
    model: str | None = typer.Option(default=None, help="Override Whisper model for this file."),
) -> None:
    """Transcribe an existing audio file into a meeting folder.
    """

    settings = Settings()
    meeting = create_meeting(settings.output_dir(), title)
    meeting.directory.mkdir(parents=True, exist_ok=True)

    console.print(f"Transcribing {audio_file}...")
    transcriber = ChunkTranscriber(settings, model_name=model, language=language)
    segments = transcriber.transcribe_file(audio_file)
    transcript = render_transcript(
        title=meeting.title,
        started_at=meeting.started_at,
        duration_seconds=None,
        model_name=transcriber.model_name,
        device=settings.whisper_device,
        compute_type=transcriber.compute_type,
        segments=segments,
    )
    meeting.transcript_md.write_text(transcript, encoding="utf-8")

    write_metadata(
        meeting,
        {
            "title": meeting.title,
            "started_at": meeting.started_at.isoformat(),
            "source_audio": str(audio_file),
            "whisper_model": transcriber.model_name,
            "device": settings.whisper_device,
            "compute_type": transcriber.compute_type,
            "language": language or settings.whisper_language,
        },
    )

    if summary:
        summary_text = generate_summary(settings, transcript, meeting.title)
        meeting.summary_md.write_text(summary_text, encoding="utf-8")

    console.print(f"Saved outputs to {meeting.directory}")


@app.command()
def record(
    title: str | None = typer.Option(default=None, help="Optional meeting title."),
    summary: bool = typer.Option(default=False, help="Generate summary after recording stops."),
) -> None:
    """Record mic + system audio and show live transcription.
    """

    settings = Settings()
    if not command_exists("ffmpeg"):
        raise typer.BadParameter("ffmpeg is not installed. Install it with: sudo apt install ffmpeg")
    if not settings.audio_mic_source or not settings.audio_system_source:
        raise typer.BadParameter("Set AUDIO_MIC_SOURCE and AUDIO_SYSTEM_SOURCE in .env. Run `transcribe devices` first.")

    settings.output_dir().mkdir(parents=True, exist_ok=True)
    session = LiveSession(settings, title, summary)

    recorder = start_recording(
        paths=session.paths,
        mic_source=settings.audio_mic_source,
        system_source=settings.audio_system_source,
        chunk_seconds=settings.chunk_seconds,
    )

    worker = threading.Thread(target=session.process_chunk_files, daemon=True)
    worker.start()

    start_time = time.time()
    try:
        with raw_terminal_input(), Live(session.render(0), console=console, refresh_per_second=4) as live:
            while recorder.process.poll() is None and not session._stop_event.is_set():
                elapsed = time.time() - start_time
                live.update(session.render(elapsed))

                key = read_key()
                if key and key.lower() == "q":
                    session.status = "Stopping recorder"
                    live.update(session.render(elapsed))
                    break

                time.sleep(0.25)
    except KeyboardInterrupt:
        session.status = "Emergency stop requested"
    finally:
        return_code = recorder.stop()
        session._stop_event.set()
        session.status = "Finalizing transcript"
        worker.join()

    if session._error:
        console.print(session._error)
        raise typer.Exit(code=1)
    if return_code != 0:
        stderr = recorder.process.stderr.read() if recorder.process.stderr else ""
        console.print(stderr.strip() or "ffmpeg exited with an error")
        raise typer.Exit(code=1)

    duration_seconds = time.time() - start_time
    session.write_transcript(duration_seconds)

    console.print("Saving MP3...")
    conversion = convert_wav_to_mp3(session.paths.raw_wav, session.paths.recording_mp3)
    if conversion.returncode != 0:
        console.print(conversion.stderr.strip() or "Unable to convert recording to MP3")
        raise typer.Exit(code=1)

    transcript_markdown = session.paths.transcript_md.read_text(encoding="utf-8")
    if summary:
        console.print("Generating summary...")
        summary_text = generate_summary(settings, transcript_markdown, session.paths.title)
        session.paths.summary_md.write_text(summary_text, encoding="utf-8")

    write_metadata(
        session.paths,
        {
            "title": session.paths.title,
            "started_at": session.paths.started_at.isoformat(),
            "duration_seconds": duration_seconds,
            "whisper_model": settings.whisper_model,
            "device": settings.whisper_device,
            "compute_type": settings.whisper_compute_type,
            "language": settings.whisper_language,
            "audio_sources": {
                "mic": settings.audio_mic_source,
                "system": settings.audio_system_source,
            },
        },
    )

    if session.paths.raw_wav.exists():
        session.paths.raw_wav.unlink()
    if session.paths.chunks_dir.exists():
        shutil.rmtree(session.paths.chunks_dir)

    console.print(f"Saved meeting folder: {session.paths.directory}")


@app.callback()
def main() -> None:
    """Transcribe meetings locally on Linux.
    """


if __name__ == "__main__":
    app()
