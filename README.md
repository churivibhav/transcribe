# transcribe

Local-first Linux meeting recorder and multilingual transcription tool powered by `faster-whisper`, with optional WhisperLive/OpenVINO transcription.

## Features

- Records microphone audio and system audio into one mixed meeting recording.
- Shows live transcription in the terminal while the meeting is running.
- Supports engine selection at startup and engine switching during recording.
- Saves each meeting into its own folder named from title plus datetime.
- Writes `recording.mp3`, `transcript.md`, `metadata.json`, and optional `summary.md`.
- Uses local transcription by default and can optionally generate meeting notes with OpenAI.

## Requirements

- Linux with PulseAudio or PipeWire Pulse compatibility.
- Python 3.11+
- `ffmpeg`
- `pactl` from `pulseaudio-utils`

On Zorin OS / Ubuntu:

```bash
sudo apt install ffmpeg pulseaudio-utils python3-venv
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
cp .env.example .env
```

## Configure Audio Sources

List available sources:

```bash
transcribe devices
```

Then update `.env`:

```env
TRANSCRIBE_OUTPUT_DIR=~/Meetings
TRANSCRIBE_ENGINE=faster-whisper

WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_CPU_THREADS=4
WHISPER_NUM_WORKERS=1
WHISPER_LANGUAGE=auto

WHISPERLIVE_URL=http://127.0.0.1:9090
WHISPERLIVE_MODEL=small
WHISPERLIVE_LANGUAGE=auto

CHUNK_SECONDS=10

AUDIO_MIC_SOURCE=alsa_input.pci-0000_00_1f.3.analog-stereo
AUDIO_SYSTEM_SOURCE=alsa_output.pci-0000_00_1f.3.analog-stereo.monitor

SUMMARY_PROVIDER=none
OPENAI_API_KEY=
OPENAI_SUMMARY_MODEL=gpt-4o-mini
```

## Usage

Check environment and config:

```bash
transcribe doctor
```

Start live meeting recording:

```bash
transcribe record
transcribe record --title "Weekly Sync"
transcribe record --title "Weekly Sync" --summary
transcribe record --engine openvino
```

During recording, press `1` to use `faster-whisper`, `2` to use `openvino`, or `q` to stop and save. Switching engines affects future audio only; already transcribed audio is left unchanged.

`faster-whisper` uses completed audio chunks. Live transcript updates appear after each completed chunk plus model processing time. `CHUNK_SECONDS=10` is the default balance. Use `5` for faster updates or `20` for more context and fewer chunks.

`openvino` streams mixed audio directly to WhisperLive over WebSocket while recording. It does not wait for chunk files to complete, so it should update sooner than the chunked `faster-whisper` path. The app still writes chunk files internally so you can switch back to `faster-whisper` with `1`.

For OpenVINO streaming, continuous speech is kept on the same timestamped transcript line and updated in place as WhisperLive revises partial text. A new timestamped line is started after a pause.

Verify configured audio sources before a meeting:

```bash
transcribe test-audio --source mic --seconds 8
transcribe test-audio --source system --seconds 8
transcribe test-audio --source both --seconds 8
```

Each command saves a short MP3 under `TRANSCRIBE_OUTPUT_DIR`. Play it to confirm the selected source contains the expected audio.

Transcribe an existing file:

```bash
transcribe transcribe-file ./meeting.mp3 --title "Imported Meeting"
transcribe transcribe-file ./meeting.mp3 --title "Imported Meeting" --engine openvino
```

For a separate retry transcript with a language hint or larger model:

```bash
transcribe transcribe-file ./meeting.mp3 --title "Marathi Retry" --language mr --model medium
```

Stop a live meeting by pressing `q` in the terminal UI. Use `Ctrl+C` only as an emergency abort.

## OpenVINO Engine

The `openvino` engine uses a local WhisperLive WebSocket server. This is optional; the default `faster-whisper` engine does not need it.

Install the extra system and Python dependencies:

```bash
sudo apt install portaudio19-dev
pip install whisper-live
```

Start the WhisperLive OpenVINO server in a separate terminal and keep it running while recording:

```bash
transcribe openvino-server
```

The server listens on `ws://127.0.0.1:9090`, matching this `.env` setting:

```env
WHISPERLIVE_URL=ws://127.0.0.1:9090
WHISPERLIVE_MODEL=OpenVINO/whisper-tiny-fp16-ov
```

Then record with OpenVINO:

```bash
transcribe record --engine openvino
```

Or start with the default engine and switch during recording with `2`. Press `1` to switch back to the local `faster-whisper` engine.

The first OpenVINO request may take longer because WhisperLive downloads and caches the OpenVINO model under `~/.cache/openvino_whisper_models`.

If `transcribe openvino-server` says the address is already in use, a server is already running on that port. Either use it, stop it, or start another server on a different port and update `WHISPERLIVE_URL`.

## Output Layout

With title:

```text
~/Meetings/
  weekly-sync-2026-06-28-1430/
    recording.mp3
    transcript.md
    summary.md
    metadata.json
```

Without title:

```text
~/Meetings/
  2026-06-28-1430/
    recording.mp3
    transcript.md
    metadata.json
```

## Notes

- `small` is the recommended starting model for live CPU transcription.
- `medium` may improve multilingual accuracy but can lag on CPU.
- Speaker diarization is intentionally deferred to a future version.
