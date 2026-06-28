# transcribe

Local-first Linux meeting recorder and multilingual transcription tool powered by `faster-whisper`.

## Features

- Records microphone audio and system audio into one mixed meeting recording.
- Shows live transcription in the terminal while the meeting is running.
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

WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_CPU_THREADS=4
WHISPER_NUM_WORKERS=1

CHUNK_SECONDS=20

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
```

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
```

For a separate retry transcript with a language hint or larger model:

```bash
transcribe transcribe-file ./meeting.mp3 --title "Marathi Retry" --language mr --model medium
```

Stop a live meeting by pressing `q` in the terminal UI. Use `Ctrl+C` only as an emergency abort.

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
