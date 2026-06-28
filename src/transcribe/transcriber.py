from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Protocol
from urllib.parse import urlparse
import uuid
import wave

from faster_whisper import WhisperModel
import numpy as np
import websocket

from .config import Settings
from .markdown import TranscriptSegment

DEFAULT_ENGINE = "faster-whisper"
OPENVINO_ENGINE = "openvino"
SUPPORTED_ENGINES = {DEFAULT_ENGINE, OPENVINO_ENGINE}


class TranscriptionEngine(Protocol):
    name: str
    model_name: str
    device: str
    compute_type: str

    def transcribe_file(self, audio_path: Path, offset_seconds: float = 0.0) -> list[TranscriptSegment]:
        ...


class FasterWhisperEngine:
    name = DEFAULT_ENGINE

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
        self.device = settings.whisper_device
        self.language = _normalize_language(language or settings.whisper_language)
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
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


class WhisperLiveOpenVINOEngine:
    name = OPENVINO_ENGINE
    device = "openvino"
    compute_type = "websocket"

    def __init__(self, settings: Settings, model_name: str | None = None, language: str | None = None) -> None:
        self._settings = settings
        self.websocket_url = _normalize_whisperlive_websocket_url(settings.whisperlive_url)
        self.model_name = model_name or settings.whisperlive_model
        self.language = _normalize_language(language or settings.whisperlive_language)

    def transcribe_file(self, audio_path: Path, offset_seconds: float = 0.0) -> list[TranscriptSegment]:
        prepared_audio_path = _ensure_pcm_wav(audio_path)
        uid = str(uuid.uuid4())
        ws = websocket.create_connection(self.websocket_url, timeout=10)
        segments: list[TranscriptSegment] = []
        try:
            ws.send(
                json.dumps(
                    {
                        "uid": uid,
                        "language": self.language,
                        "task": "transcribe",
                        "model": self.model_name,
                        "use_vad": False,
                        "send_last_n_segments": 10,
                        "no_speech_thresh": 0.45,
                        "clip_audio": False,
                        "same_output_threshold": 1,
                        "enable_translation": False,
                        "target_language": "en",
                        "hotwords": None,
                        "enable_diarization": False,
                        "max_speakers": 10,
                        "word_timestamps": False,
                    }
                )
            )
            self._wait_for_server_ready(ws, uid)
            for packet in _wav_file_to_float32_packets(prepared_audio_path):
                ws.send(packet, opcode=websocket.ABNF.OPCODE_BINARY)
            deadline = time.time() + 60
            last_segments: list[dict] = []
            while time.time() < deadline:
                try:
                    raw_message = ws.recv()
                except websocket.WebSocketTimeoutException:
                    break
                if not raw_message:
                    break
                message = json.loads(raw_message)
                if message.get("uid") != uid:
                    continue
                if message.get("status") == "ERROR":
                    raise RuntimeError(str(message.get("message", "WhisperLive server error")))
                if message.get("segments"):
                    last_segments = message["segments"]
                    if any(segment.get("completed") for segment in last_segments):
                        break
            ws.send(b"END_OF_AUDIO", opcode=websocket.ABNF.OPCODE_BINARY)
            segments = _segments_from_whisperlive_segments(last_segments, offset_seconds)
        finally:
            ws.close()
            if prepared_audio_path != audio_path and prepared_audio_path.exists():
                prepared_audio_path.unlink()
        return segments

    def open_stream(self, on_segments, offset_seconds: float = 0.0) -> WhisperLiveOpenVINOStream:
        stream = WhisperLiveOpenVINOStream(self, on_segments, offset_seconds)
        stream.start()
        return stream

    def _wait_for_server_ready(self, ws: websocket.WebSocket, uid: str) -> None:
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                raw_message = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not raw_message:
                continue
            message = json.loads(raw_message)
            if message.get("uid") != uid:
                continue
            if message.get("status") == "ERROR":
                raise RuntimeError(str(message.get("message", "WhisperLive server error")))
            if message.get("message") == "SERVER_READY":
                backend = message.get("backend")
                if backend != "openvino":
                    raise RuntimeError(f"WhisperLive server is using backend {backend!r}, expected 'openvino'")
                return
        raise TimeoutError("Timed out waiting for WhisperLive OpenVINO server")


class WhisperLiveOpenVINOStream:
    def __init__(self, engine: WhisperLiveOpenVINOEngine, on_segments, offset_seconds: float) -> None:
        self._engine = engine
        self._on_segments = on_segments
        self._offset_seconds = offset_seconds
        self._uid = str(uuid.uuid4())
        self._ws: websocket.WebSocket | None = None
        self._stop_event = threading.Event()
        self._receiver: threading.Thread | None = None
        self._send_lock = threading.Lock()

    def start(self) -> None:
        ws = websocket.create_connection(self._engine.websocket_url, timeout=10)
        self._ws = ws
        ws.send(
            json.dumps(
                {
                    "uid": self._uid,
                    "language": self._engine.language,
                    "task": "transcribe",
                    "model": self._engine.model_name,
                    "use_vad": False,
                    "send_last_n_segments": 10,
                    "no_speech_thresh": 0.45,
                    "clip_audio": False,
                    "same_output_threshold": 1,
                    "enable_translation": False,
                    "target_language": "en",
                    "hotwords": None,
                    "enable_diarization": False,
                    "max_speakers": 10,
                    "word_timestamps": False,
                }
            )
        )
        self._engine._wait_for_server_ready(ws, self._uid)
        self._receiver = threading.Thread(target=self._receive_loop, daemon=True)
        self._receiver.start()

    def send(self, packet: bytes) -> None:
        if self._stop_event.is_set() or self._ws is None:
            return
        with self._send_lock:
            self._ws.send(packet, opcode=websocket.ABNF.OPCODE_BINARY)

    def close(self) -> None:
        self._stop_event.set()
        if self._ws is not None:
            try:
                with self._send_lock:
                    self._ws.send(b"END_OF_AUDIO", opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception:
                pass
            try:
                self._ws.close()
            except Exception:
                pass
        if self._receiver is not None:
            self._receiver.join(timeout=5)

    def _receive_loop(self) -> None:
        if self._ws is None:
            return
        while not self._stop_event.is_set():
            try:
                raw_message = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                break
            if not raw_message:
                continue
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                continue
            if message.get("uid") != self._uid:
                continue
            if message.get("segments"):
                self._on_segments(_segments_from_whisperlive_segments(message["segments"], self._offset_seconds))


def create_transcription_engine(
    settings: Settings,
    engine_name: str | None = None,
    model_name: str | None = None,
    compute_type: str | None = None,
    language: str | None = None,
) -> TranscriptionEngine:
    engine = normalize_engine_name(engine_name or settings.transcribe_engine)
    if engine == DEFAULT_ENGINE:
        return FasterWhisperEngine(settings, model_name=model_name, compute_type=compute_type, language=language)
    if engine == OPENVINO_ENGINE:
        return WhisperLiveOpenVINOEngine(settings, model_name=model_name, language=language)
    raise ValueError(f"Unsupported transcription engine: {engine}")


def normalize_engine_name(engine_name: str | None) -> str:
    cleaned = (engine_name or DEFAULT_ENGINE).strip().lower().replace("_", "-")
    aliases = {
        "faster": DEFAULT_ENGINE,
        "whisper": DEFAULT_ENGINE,
        "faster-whisper": DEFAULT_ENGINE,
        "whisperlive": OPENVINO_ENGINE,
        "whisperlive-openvino": OPENVINO_ENGINE,
        "openvino": OPENVINO_ENGINE,
    }
    engine = aliases.get(cleaned, cleaned)
    if engine not in SUPPORTED_ENGINES:
        supported = ", ".join(sorted(SUPPORTED_ENGINES))
        raise ValueError(f"Unsupported engine '{engine_name}'. Supported engines: {supported}")
    return engine


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_whisperlive_websocket_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme in {"ws", "wss"}:
        return value.rstrip("/")
    if parsed.scheme in {"http", "https"}:
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return parsed._replace(scheme=scheme).geturl().rstrip("/")
    return f"ws://{value.strip().rstrip('/')}"


def _wav_file_to_float32_packets(audio_path: Path, frame_count: int = 4096):
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if sample_width != 2:
            raise ValueError("WhisperLive OpenVINO engine expects 16-bit PCM WAV chunks")
        while True:
            data = wav_file.readframes(frame_count)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.int16)
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
            yield (samples.astype(np.float32) / 32768.0).tobytes()


def _ensure_pcm_wav(audio_path: Path) -> Path:
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    output = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or f"Unable to convert {audio_path} to WAV")
    return output


def _segments_from_whisperlive_segments(raw_segments: list[dict], offset_seconds: float) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for raw_segment in raw_segments:
        text = str(raw_segment.get("text", "")).strip()
        if not text:
            continue
        start = _safe_float(raw_segment.get("start"), 0.0)
        end = _safe_float(raw_segment.get("end"), start)
        segments.append(
            TranscriptSegment(
                start_seconds=offset_seconds + start,
                end_seconds=offset_seconds + end,
                text=text,
            )
        )
    return segments


def _normalize_language(language: str | None) -> str | None:
    if not language:
        return None
    cleaned = language.strip().lower()
    if cleaned in {"", "auto", "detect"}:
        return None
    return cleaned
