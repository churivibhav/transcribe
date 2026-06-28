from __future__ import annotations

from pathlib import Path
import importlib.util
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
from .transcriber import DEFAULT_ENGINE, OPENVINO_ENGINE, create_transcription_engine, normalize_engine_name

app = typer.Typer(help="Linux meeting recorder and live transcription tool.")
console = Console()


class LiveSession:
    def __init__(self, settings: Settings, title: str | None, with_summary: bool, engine_name: str | None) -> None:
        self.settings = settings
        self.paths = create_meeting(settings.output_dir(), title)
        self.with_summary = with_summary
        self.segments: list[TranscriptSegment] = []
        self.engine = create_transcription_engine(settings, engine_name=engine_name)
        self.initial_engine_name = self.engine.name
        self.engine_switches: list[dict[str, float | str]] = []
        self._stream_segments: dict[float, TranscriptSegment] = {}
        self._stop_event = threading.Event()
        self._error: str | None = None
        self._lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self.status = "Recording"

    def append_segments(self, new_segments: list[TranscriptSegment]) -> None:
        with self._lock:
            self.segments.extend(new_segments)
            self.segments.sort(key=lambda segment: segment.start_seconds)
        self.write_transcript(None)

    def update_stream_segments(self, new_segments: list[TranscriptSegment]) -> None:
        grouped_segments = self._group_stream_segments(new_segments)
        with self._lock:
            for segment in grouped_segments:
                key = self._stream_segment_key(segment)
                self._stream_segments[key] = segment
            chunk_segments = [
                segment
                for segment in self.segments
                if self._stream_segment_key(segment) not in self._stream_segments
            ]
            self.segments = sorted(chunk_segments + list(self._stream_segments.values()), key=lambda segment: segment.start_seconds)
        self.write_transcript(None)

    @staticmethod
    def _stream_segment_key(segment: TranscriptSegment) -> float:
        return round(segment.start_seconds, 1)

    @staticmethod
    def _group_stream_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        if not segments:
            return []

        sorted_segments = sorted(segments, key=lambda segment: segment.start_seconds)
        grouped: list[TranscriptSegment] = []
        current_start = sorted_segments[0].start_seconds
        current_end = sorted_segments[0].end_seconds
        current_text_parts = [sorted_segments[0].text.strip()]

        for segment in sorted_segments[1:]:
            text = segment.text.strip()
            if not text:
                continue
            gap_seconds = segment.start_seconds - current_end
            if gap_seconds <= 2.0:
                current_end = max(current_end, segment.end_seconds)
                if not current_text_parts or current_text_parts[-1] != text:
                    current_text_parts.append(text)
                continue

            grouped.append(
                TranscriptSegment(
                    start_seconds=current_start,
                    end_seconds=current_end,
                    text=" ".join(current_text_parts).strip(),
                )
            )
            current_start = segment.start_seconds
            current_end = segment.end_seconds
            current_text_parts = [text]

        grouped.append(
            TranscriptSegment(
                start_seconds=current_start,
                end_seconds=current_end,
                text=" ".join(current_text_parts).strip(),
            )
        )
        return grouped

    def write_transcript(self, duration_seconds: float | None) -> None:
        with self._lock:
            segments = list(self.segments)
        with self._engine_lock:
            engine = self.engine
        content = render_transcript(
            title=self.paths.title,
            started_at=self.paths.started_at,
            duration_seconds=duration_seconds,
            model_name=engine.model_name,
            device=engine.device,
            compute_type=engine.compute_type,
            segments=segments,
        )
        self.paths.transcript_md.write_text(content, encoding="utf-8")

    def switch_engine(self, engine_name: str, elapsed_seconds: float) -> None:
        next_engine_name = normalize_engine_name(engine_name)
        with self._engine_lock:
            if self.engine.name == next_engine_name:
                self.status = f"Already using {next_engine_name}"
                return
        next_engine = create_transcription_engine(self.settings, engine_name=next_engine_name)
        with self._engine_lock:
            self.engine = next_engine
            self.engine_switches.append(
                {
                    "timestamp_seconds": elapsed_seconds,
                    "engine": next_engine.name,
                }
            )
        self.status = f"Switched to {next_engine.name}"

    def render(self, elapsed_seconds: float) -> Panel:
        with self._engine_lock:
            engine = self.engine
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_row(f"Title: {self.paths.title}")
        grid.add_row(f"Elapsed: {seconds_to_clock(elapsed_seconds)}")
        grid.add_row(f"Status: {self.status}")
        grid.add_row(f"Engine: {engine.name}")
        grid.add_row(f"Model: {engine.model_name} / {engine.device} / {engine.compute_type}")
        grid.add_row(f"Output: {self.paths.directory}")
        grid.add_row("Keys: 1 faster-whisper | 2 openvino | q stop/save. Ctrl+C is emergency abort.")

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
                    with self._engine_lock:
                        engine = self.engine
                    if engine.name == OPENVINO_ENGINE:
                        seen.add(chunk_file)
                        continue
                    new_segments = engine.transcribe_file(chunk_file, offset_seconds=offset)
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
                with self._engine_lock:
                    engine = self.engine
                if engine.name == OPENVINO_ENGINE:
                    seen.add(chunk_file)
                    continue
                new_segments = engine.transcribe_file(chunk_file, offset_seconds=offset)
                seen.add(chunk_file)
                if new_segments:
                    self.append_segments(new_segments)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self._error = str(exc)


class AudioStreamRouter:
    def __init__(self, stdout, session: LiveSession) -> None:
        self._stdout = stdout
        self._session = session
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._openvino_stream = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.deactivate_openvino()
        self._thread.join(timeout=10)

    def activate_openvino(self, offset_seconds: float) -> None:
        with self._lock:
            if self._openvino_stream is not None:
                return
            with self._session._engine_lock:
                engine = self._session.engine
            if engine.name != OPENVINO_ENGINE or not hasattr(engine, "open_stream"):
                return
            self._openvino_stream = engine.open_stream(self._session.update_stream_segments, offset_seconds=offset_seconds)

    def deactivate_openvino(self) -> None:
        with self._lock:
            stream = self._openvino_stream
            self._openvino_stream = None
        if stream is not None:
            stream.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            packet = self._stdout.read(16384)
            if not packet:
                break
            with self._lock:
                stream = self._openvino_stream
            if stream is not None:
                try:
                    stream.send(packet)
                except Exception as exc:  # pragma: no cover - runtime streaming path
                    self._session._error = str(exc)
                    self._session._stop_event.set()
                    break


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
        ("whisper-live", importlib.util.find_spec("whisper_live") is not None),
    ]

    table = Table(title="Doctor")
    table.add_column("Check")
    table.add_column("Status")
    for name, ok in rows:
        table.add_row(name, "ok" if ok else "missing")
    console.print(table)

    console.print(f"Output directory: {settings.output_dir()}")
    console.print(f"Default engine: {normalize_engine_name(settings.transcribe_engine)}")
    console.print(f"WhisperLive URL: {settings.whisperlive_url}")
    if not settings.audio_mic_source or not settings.audio_system_source:
        console.print("Audio sources are not fully configured in .env")
    if settings.summary_enabled() and not settings.openai_api_key:
        console.print("SUMMARY_PROVIDER is enabled but OPENAI_API_KEY is missing")


@app.command("openvino-server")
def openvino_server(
    host: str = typer.Option(default="127.0.0.1", help="Host for the WhisperLive WebSocket server."),
    port: int = typer.Option(default=9090, help="Port for the WhisperLive WebSocket server."),
    rest_port: int = typer.Option(default=8000, help="Optional REST API port exposed by WhisperLive."),
) -> None:
    """Start the WhisperLive OpenVINO server used by `--engine openvino`.
    """

    try:
        from whisper_live.server import TranscriptionServer
    except ImportError as exc:
        raise typer.BadParameter("whisper-live is not installed. Install it with: pip install whisper-live") from exc

    console.print(f"Starting WhisperLive OpenVINO server on ws://{host}:{port}")
    console.print(f"REST compatibility endpoint, if needed, will listen on http://{host}:{rest_port}")
    TranscriptionServer().run(
        host,
        port=port,
        backend="openvino",
        rest_port=rest_port,
        enable_rest=True,
        single_model=True,
        max_clients=2,
        max_connection_time=600,
    )


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
    engine: str | None = typer.Option(default=None, help="Transcription engine: faster-whisper or openvino."),
) -> None:
    """Transcribe an existing audio file into a meeting folder.
    """

    settings = Settings()
    meeting = create_meeting(settings.output_dir(), title)
    meeting.directory.mkdir(parents=True, exist_ok=True)

    console.print(f"Transcribing {audio_file}...")
    transcriber = create_transcription_engine(settings, engine_name=engine, model_name=model, language=language)
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
            "engine": transcriber.name,
            "whisper_model": transcriber.model_name,
            "device": transcriber.device,
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
    engine: str | None = typer.Option(default=None, help="Transcription engine: faster-whisper or openvino."),
) -> None:
    """Record mic + system audio and show live transcription.
    """

    settings = Settings()
    if not command_exists("ffmpeg"):
        raise typer.BadParameter("ffmpeg is not installed. Install it with: sudo apt install ffmpeg")
    if not settings.audio_mic_source or not settings.audio_system_source:
        raise typer.BadParameter("Set AUDIO_MIC_SOURCE and AUDIO_SYSTEM_SOURCE in .env. Run `transcribe devices` first.")

    settings.output_dir().mkdir(parents=True, exist_ok=True)
    session = LiveSession(settings, title, summary, engine)

    recorder = start_recording(
        paths=session.paths,
        mic_source=settings.audio_mic_source,
        system_source=settings.audio_system_source,
        chunk_seconds=settings.chunk_seconds,
        stream_stdout=True,
    )

    stream_router = AudioStreamRouter(recorder.process.stdout, session)
    stream_router.start()
    if session.engine.name == OPENVINO_ENGINE:
        try:
            stream_router.activate_openvino(0.0)
            session.status = "Streaming to OpenVINO"
        except Exception as exc:
            recorder.stop()
            stream_router.stop()
            raise typer.BadParameter(f"Could not start OpenVINO stream: {exc}") from exc

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
                if key == "1":
                    try:
                        session.switch_engine(DEFAULT_ENGINE, elapsed)
                        stream_router.deactivate_openvino()
                    except Exception as exc:  # pragma: no cover - runtime configuration path
                        session.status = f"Could not switch to {DEFAULT_ENGINE}: {exc}"
                    live.update(session.render(elapsed))
                if key == "2":
                    try:
                        session.switch_engine(OPENVINO_ENGINE, elapsed)
                        stream_router.activate_openvino(elapsed)
                        session.status = "Streaming to OpenVINO"
                    except Exception as exc:  # pragma: no cover - runtime configuration path
                        session.status = f"Could not switch to {OPENVINO_ENGINE}: {exc}"
                    live.update(session.render(elapsed))

                time.sleep(0.25)
    except KeyboardInterrupt:
        session.status = "Emergency stop requested"
    finally:
        return_code = recorder.stop()
        stream_router.stop()
        session._stop_event.set()
        session.status = "Finalizing transcript"
        worker.join()

    if session._error:
        console.print(session._error)
        raise typer.Exit(code=1)
    if return_code != 0:
        stderr = recorder.process.stderr.read().decode("utf-8", errors="replace") if recorder.process.stderr else ""
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
            "engine": session.initial_engine_name,
            "engine_switches": session.engine_switches,
            "whisper_model": session.engine.model_name,
            "device": session.engine.device,
            "compute_type": session.engine.compute_type,
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
