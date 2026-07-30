# Voice-Activated Speech To Text

An offline, hands-free speech-to-text pipeline built in C++17 for macOS and Apple Silicon. It uses `whisper.cpp` for transcription, RtAudio for microphone capture, and a background worker architecture so audio collection stays responsive while inference runs separately.

The v2 example continuously listens, calibrates itself to room noise, detects speech with a lightweight voice activity detector, and emits transcriptions when you naturally pause.

## Highlights

- **Continuous hands-free capture** with no start/stop hotkey required.
- **Self-calibrating VAD** that samples ambient noise during startup and derives a dynamic activation threshold.
- **Pre-roll buffering** to avoid clipping the first syllables of a phrase.
- **Background inference worker** based on the local `cBaseWorker_V2` lifecycle class.
- **Apple Silicon-friendly build** using Homebrew `whisper-cpp`, RtAudio, GGML, and Apple acceleration frameworks.
- **On-the-fly resampling** from common microphone rates such as 48 kHz down to Whisper's required 16 kHz mono PCM.

## Project Layout

```text
SpeechToText/
├── CMakeLists.txt          # v2 executable build: WhisperSTTWorkerV2
├── CMakeLists_v1.txt       # legacy/simple STT example build
├── README.md
├── .gitignore
├── example2_stt.cpp        # continuous voice-activated demo
├── stt_whisper_v2.hpp      # worker-based STT public interface
├── stt_whisper_v2.cpp      # VAD, resampling, capture, and inference
├── term_util.h             # raw terminal polling helper
├── example_stt.cpp         # legacy keyboard-triggered demo
├── stt_whisper.hpp
└── stt_whisper.cpp
```

The v2 implementation also expects the sibling `ThreadComponent` project to be available at:

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
brew install cmake pkg-config rtaudio whisper-cpp
```

## Model Setup

Download a Whisper GGML model before running the example. The current demo expects:

```text
/Users/phelelanicwele/models/ggml-base.en.bin
```

Create the folder and download the base English model:

```bash
mkdir -p ~/models
curl -L \
  -o ~/models/ggml-base.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

To use a different model path, update `model_path` in `example2_stt.cpp`.

## Build

From this directory:

```bash
cmake -S . -B build
cmake --build build
```

The executable is created at:

```text
build/WhisperSTTWorkerV2
```

## Run

```bash
./build/WhisperSTTWorkerV2
```

When the app starts, stay quiet during the initial calibration window. After calibration, speak naturally. When you pause, the current phrase is queued for Whisper inference and printed as a transcription event.

Press `Q` to stop the listener, close audio streams, join the worker thread, and exit cleanly.

Example output:

```text
[VAD Calibration Complete] Ambient noise: 3.12145e-06 -> Dynamic Threshold auto-set to: 0.003

[TRANSCRIPTION EVENT]: Are you missing any of my words?
=========================================================
```

## Public Interface

Downstream code can use the worker class with an optional callback:

```cpp
class SpeechToText : public cBaseWorker_V2
{
public:
    using TranscriptionCallback = std::function<void(const std::string &text)>;

    SpeechToText(const std::string &model_path, TranscriptionCallback callback = nullptr);
    ~SpeechToText() noexcept override;

    void set_callback(TranscriptionCallback callback);

protected:
    bool preRun() override;
    void run() override;
    void stopTriggered() override;
};
```

Minimal usage:

```cpp
auto on_transcription = [](const std::string &text) {
    std::cout << text << std::endl;
};

SpeechToText stt("~/models/ggml-base.en.bin", on_transcription);
stt.startThread(cBaseWorker_V2::duration_type{15000});
```

## Troubleshooting

### CMake cannot find RtAudio or Whisper

Make sure Homebrew packages are installed and visible to `pkg-config`:

```bash
brew install pkg-config rtaudio whisper-cpp
pkg-config --modversion rtaudio
pkg-config --modversion whisper
```

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
- Expose more VAD timing and threshold parameters through a small configuration object.
