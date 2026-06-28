from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from .config import Settings
from .markdown import TranscriptSegment


class ChunkTranscriber:
    def __init__(
        self,
        settings: Settings,
        model_name: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
    ) -> None:
        self._settings = settings
        self.model_name = model_name or settings.whisper_model
        self.compute_type = compute_type or settings.whisper_compute_type
        self.language = _normalize_language(language or settings.whisper_language)
        self._model = WhisperModel(
            self.model_name,
            device=settings.whisper_device,
            compute_type=self.compute_type,
            cpu_threads=settings.whisper_cpu_threads,
            num_workers=settings.whisper_num_workers,
        )

    def transcribe_file(self, audio_path: Path, offset_seconds: float = 0.0) -> list[TranscriptSegment]:
        segments, _ = self._model.transcribe(
            str(audio_path),
            vad_filter=True,
            language=self.language,
            task="transcribe",
            beam_size=5,
        )

        transcript_segments: list[TranscriptSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            transcript_segments.append(
                TranscriptSegment(
                    start_seconds=offset_seconds + float(segment.start),
                    end_seconds=offset_seconds + float(segment.end),
                    text=text,
                )
            )
        return transcript_segments


def _normalize_language(language: str | None) -> str | None:
    if not language:
        return None
    cleaned = language.strip().lower()
    if cleaned in {"", "auto", "detect"}:
        return None
    return cleaned
