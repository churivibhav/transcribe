from __future__ import annotations

from dataclasses import dataclass
import subprocess
from pathlib import Path

from .meeting import MeetingPaths


@dataclass(slots=True)
class RecorderProcess:
    process: subprocess.Popen[str]
    chunk_pattern: Path

    def stop(self) -> int:
        if self.process.poll() is None:
            if self.process.stdin:
                try:
                    self.process.stdin.write("q\n")
                    self.process.stdin.flush()
                except BrokenPipeError:
                    pass
            try:
                return self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    return self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    return self.process.wait(timeout=10)
        return self.process.returncode or 0


def start_recording(paths: MeetingPaths, mic_source: str, system_source: str, chunk_seconds: int) -> RecorderProcess:
    paths.directory.mkdir(parents=True, exist_ok=True)
    chunk_pattern = paths.chunks_dir / "chunk_%05d.wav"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "pulse",
        "-i",
        mic_source,
        "-f",
        "pulse",
        "-i",
        system_source,
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0,asplit=2[full][chunks]",
        "-map",
        "[full]",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(paths.raw_wav),
        "-map",
        "[chunks]",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(chunk_pattern),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return RecorderProcess(process=process, chunk_pattern=chunk_pattern)


def convert_wav_to_mp3(raw_wav: Path, output_mp3: Path) -> subprocess.CompletedProcess[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_wav),
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_mp3),
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def record_audio_sample(
    output_wav: Path,
    mic_source: str,
    system_source: str,
    seconds: int,
    source: str,
) -> subprocess.CompletedProcess[str]:
    if source == "mic":
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "pulse",
            "-i",
            mic_source,
            "-t",
            str(seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(output_wav),
        ]
    elif source == "system":
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "pulse",
            "-i",
            system_source,
            "-t",
            str(seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(output_wav),
        ]
    else:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "pulse",
            "-i",
            mic_source,
            "-f",
            "pulse",
            "-i",
            system_source,
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0[mixed]",
            "-map",
            "[mixed]",
            "-t",
            str(seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(output_wav),
        ]

    return subprocess.run(command, check=False, capture_output=True, text=True)


def analyze_audio_volume(audio_file: Path) -> subprocess.CompletedProcess[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_file),
        "-af",
        "volumedetect",
        "-f",
        "null",
        "/dev/null",
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True)
