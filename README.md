# Voice-Activated Speech To Text

An offline, hands-free speech-to-text pipeline built in C++17 for macOS and Apple Silicon. It uses `whisper.cpp` for transcription, RtAudio for microphone capture, and a background worker architecture so audio collection stays responsive while inference runs separately.

The v2 example continuously listens, calibrates itself to room noise, detects speech with a lightweight voice activity detector, and emits transcriptions when you naturally pause.

## Highlights

- **Continuous hands-free capture** with no start/stop hotkey required.
- **Self-calibrating VAD** that samples ambient noise during startup and derives a dynamic activation threshold (percentile-based, robust to transient noise spikes).
- **Pre-roll buffering** to avoid clipping the first syllables of a phrase.
- **Hysteresis** so brief energy dips don't chatter the speech state.
- **Background inference worker** based on the local `cBaseWorker_V2` lifecycle class — the worker joins before teardown, so no use-after-free of the Whisper context.
- **Apple Silicon-friendly build** using Homebrew `whisper-cpp`, RtAudio, GGML, and Apple acceleration frameworks.
- **On-the-fly resampling** from common microphone rates such as 48 kHz down to Whisper's required 16 kHz mono PCM, performed on the worker thread (never inside the audio callback).

## Project Layout

```text
SpeechToText/
├── CMakeLists.txt            # primary build (v2 worker + optional v1 + tests)
├── README.md
├── .gitignore
├── stt_config.hpp            # SttConfig: all tunables in one struct
├── audio_utils.hpp           # int16->float, linear resampler, text filtering
├── vad.hpp / vad.cpp         # pure-logic voice activity detector (unit-tested)
├── whisper_engine.hpp/cpp    # whisper.cpp context lifecycle + transcription
├── audio_capture.hpp/cpp     # RtAudio microphone capture wrapper
├── stt_whisper_v2.hpp/cpp    # SpeechToTextV2: continuous VAD worker
├── stt_whisper.hpp/cpp       # SpeechToTextV1: legacy push-to-talk worker
├── term_util.h               # RAII raw-terminal polling helper
├── example2_stt.cpp          # continuous voice-activated demo (default)
├── example_stt.cpp           # legacy keyboard-triggered demo (optional)
└── tests/
    └── test_audio_utils.cpp  # unit tests for resampler + VAD
```

The implementation also expects the sibling `ThreadComponent` project to be available at:

```text
../ThreadComponent/cBaseWorker_V2.h
```

## Requirements

- macOS 14 or newer
- AppleClang with C++17 support
- Homebrew
- Apple Silicon is recommended for best performance

Install the native dependencies:

```bash
brew install cmake pkg-config rtaudio whisper-cpp ggml
```

## Model Setup

Download a Whisper GGML model before running the example. The current demo defaults to:

```text
~/models/ggml-base.en.bin
```

Create the folder and download the base English model:

```bash
mkdir -p ~/models
curl -L \
  -o ~/models/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

The model path is resolved in order of: **CLI argument** (`./WhisperSTTWorkerV2 /path/to/model.bin`) → **`STT_MODEL` env var** → `~/models/ggml-base.en.bin`. A leading `~` is expanded to your home directory.

## Build

From this directory:

```bash
cmake -S . -B build
cmake --build build
```

Targets:

| Target | Description |
| --- | --- |
| `WhisperSTTWorkerV2` | Continuous VAD example (default, always built) |
| `example_stt` | Legacy push-to-talk demo (always built) |
| `stt_tests` | Unit tests (resampler + VAD, no audio hardware needed) |

The executable is created at `build/WhisperSTTWorkerV2`.

## Run

```bash
./build/WhisperSTTWorkerV2
```

When the app starts, stay quiet during the initial calibration window. After calibration, speak naturally. When you pause, the current phrase is queued for Whisper inference and printed as a transcription event.

Press `Q` to stop the listener, close audio streams, join the worker thread, and exit cleanly.

Example output:

```text
[VAD Calibration Complete] Ambient noise: 3.12145e-06 -> Dynamic Threshold auto-set to: 0.003

[TRANSCRIPTION EVENT]: Are you missing any of my words?  (latency ~340 ms)
=========================================================
```

## Configuration

Create an `SttConfig` to tune the pipeline:

```cpp
SttConfig cfg;
cfg.modelPath            = "~/models/ggml-base.en.bin";
cfg.language             = "en";
cfg.cpuThreads           = 4;
cfg.beamSize             = 5;            // 1 = greedy decoding
cfg.entropyThreshold     = 2.4f;
cfg.calibrationSeconds   = 1.5f;         // ambient-noise sampling window
cfg.pauseSeconds         = 1.2f;         // silence that ends a phrase
cfg.prerollSeconds       = 0.5f;         // audio kept before speech onset
cfg.maxPhraseSeconds     = 60.0f;        // force-slice a runaway phrase
cfg.thresholdMultiplier  = 2.5f;         // VAD threshold = floor * multiplier
cfg.hysteresisRatio      = 0.7f;         // lower bar while already speaking
cfg.thresholdFloor       = 0.003f;       // absolute minimum threshold
cfg.maxQueueDepth        = 32;           // worker back-pressure cap
cfg.bufferFrames         = 512;

SpeechToTextV2 stt(cfg, onTranscriptionComplete);
```

## Public Interface (v2)

Downstream code can use the worker class with an optional callback. The callback
receives the captured phrase's start time so callers can correlate with playback:

```cpp
class SpeechToTextV2 : public cBaseWorker_V2
{
public:
    using TranscriptionCallback =
        std::function<void(const std::string &text,
                           std::chrono::steady_clock::time_point capturedAt)>;

    SpeechToTextV2(const SttConfig &cfg, TranscriptionCallback callback = nullptr);
    ~SpeechToTextV2() noexcept override;

    void set_callback(TranscriptionCallback callback);

protected:
    bool preRun() override;
    void run() override;
    void stopTriggered() override;
};
```

Minimal usage:

```cpp
auto on_transcription = [](const std::string &text, auto /*capturedAt*/) {
    std::cout << text << std::endl;
};

SttConfig cfg;
cfg.modelPath = "~/models/ggml-base.en.bin";

SpeechToTextV2 stt(cfg, on_transcription);
stt.startThread(cBaseWorker_V2::duration_type{15000});
```

The legacy push-to-talk worker (`SpeechToTextV1`) exposes the same lifecycle plus
`start_recording()` / `stop_and_queue()`.

## Troubleshooting

### CMake cannot find RtAudio or Whisper

Make sure Homebrew packages are installed and visible to `pkg-config`:

```bash
brew install pkg-config rtaudio whisper-cpp ggml
pkg-config --modversion rtaudio
pkg-config --modversion whisper
```

CMake prefers pkg-config and falls back to `find_library` over Homebrew paths — no hard-coded Cellar versions are used.

### Model file not found

Confirm the model path exists:

```bash
ls ~/models/ggml-base.en.bin
```

### Microphone permission

macOS may require Terminal, iTerm, or your IDE to have microphone access:

```text
System Settings -> Privacy & Security -> Microphone
```

## Roadmap

- Send completed transcriptions to a local LLM runtime such as llama.cpp or Ollama.
- Connect the transcription callback to the TextToSpeech module for a full voice loop.
- Add a runtime recalibration command for changing room-noise conditions.
- Expose the existing `SttConfig` knobs through CLI flags.