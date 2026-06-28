from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    transcribe_output_dir: Path = Field(default=Path("~/Meetings"))

    whisper_model: str = Field(default="small")
    whisper_device: str = Field(default="cpu")
    whisper_compute_type: str = Field(default="int8")
    whisper_cpu_threads: int = Field(default=4)
    whisper_num_workers: int = Field(default=1)
    whisper_language: str = Field(default="auto")

    chunk_seconds: int = Field(default=20)

    audio_mic_source: str = Field(default="")
    audio_system_source: str = Field(default="")

    summary_provider: str = Field(default="none")
    openai_api_key: str = Field(default="")
    openai_summary_model: str = Field(default="gpt-4o-mini")

    def output_dir(self) -> Path:
        return self.transcribe_output_dir.expanduser().resolve()

    def summary_enabled(self) -> bool:
        return self.summary_provider.strip().lower() != "none"
