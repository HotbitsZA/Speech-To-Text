#pragma once

#include "../ThreadComponent/cBaseWorker_V2.h"
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>

// Forward declarations to avoid exposing third-party headers to downstream projects
struct whisper_context;
class RtAudio;

class SpeechToText : public cBaseWorker_V2
{
public:
    using TranscriptionCallback = std::function<void(const std::string &text)>;

    // Pass the absolute path to your ggml-*.bin model file and the callback handler
    SpeechToText(const std::string &model_path, TranscriptionCallback callback = nullptr);
    ~SpeechToText() noexcept override;

    // Trigger state changes safely from any thread
    void start_recording();
    void stop_and_queue();

    // Dynamically change or assign the text completion event listener
    void set_callback(TranscriptionCallback callback);

protected:
    // cBaseWorker_V2 Cooperative State Overrides
    bool preRun() override;
    void run() override;
    void stopTriggered() override;

private:
    static int audio_callback(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
                              double streamTime, unsigned int status, void *userData);

    // Context & Streaming Engines
    std::string m_modelPath;
    TranscriptionCallback m_onTranscriptionComplete;
    struct whisper_context *ctx = nullptr;
    std::unique_ptr<RtAudio> adc;

    // Active Capture Line Buffers
    std::vector<int16_t> audio_buffer;
    std::mutex audio_mutex;
    std::atomic<bool> is_recording{false};

    // Deep Asynchronous Execution Pipelines
    std::queue<std::vector<float>> m_taskQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;

    class Impl;
    std::unique_ptr<Impl> m_pImpl;
};


// To download the ggml-*.bin models, visit: https://github.com/ggerganov/whisper.cpp/tree/master/bin 
// Or from huggingface.co: https://huggingface.co/ggerganov/whisper.cpp/tree/main
// Or Use the following command to download the English model directly:
// wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -O ~/.cache/whisper.cpp/ggml-base.en.bin
// Download into your chosen directory, like ~/models/ggml-base.en.bin, and pass that path to the SpeechToText constructor. 
// curl -L -o ggml-base.en.bin "https://huggingface.co" 
// The ffmpeg command to convert your audio for whisper.cpp is:
// ffmpeg -i input_audio.wav -ar 16000 -ac 1 -c:a pcm_s16le output_audio.wav
// ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
/*
The Homebrew version of whisper-cpp is compiled with Metal acceleration enabled out of the box.
It leverages your Mac's graphics processors automatically.
To ensure macOS links the package resources correctly and runs at full speed,
you should paste this configuration line into your terminal:
*/
// export GGML_METAL_PATH_RESOURCES="$(brew --prefix whisper-cpp)/share/whisper-cpp"
// export DYLD_LIBRARY_PATH="/opt/homebrew/Cellar/whisper-cpp/1.9.1/lib:$DYLD_LIBRARY_PATH"
// If you installed whisper-cpp via Homebrew, the default model path is:
// ~/.cache/whisper.cpp/ggml-base.en.bin
