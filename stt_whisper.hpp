#pragma once

#include "cBaseWorker_V2.h"

#include "audio_capture.hpp"
#include "stt_config.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

class WhisperEngine;

// Legacy push-to-talk speech-to-text worker. The host calls start_recording()
// (typically while a key is held) and stop_and_queue() to hand the captured
// audio to the background transcription thread.
class SpeechToTextV1 : public cBaseWorker_V2
{
public:
    using TranscriptionCallback = std::function<void(const std::string &text)>;

    SpeechToTextV1(const SttConfig &cfg, TranscriptionCallback callback = nullptr);
    ~SpeechToTextV1() noexcept override;

    void set_callback(TranscriptionCallback callback);

    void start_recording();
    void stop_and_queue();

protected:
    bool preRun() override;
    void run() override;
    void stopTriggered() override;

private:
    void onAudioSamples(const int16_t *samples, unsigned frames);

    SttConfig m_cfg;

    mutable std::mutex m_cbMutex;
    TranscriptionCallback m_callback;

    std::unique_ptr<WhisperEngine> m_engine;
    std::unique_ptr<AudioCapture> m_capture;

    std::atomic<bool> m_recording{false};
    std::vector<int16_t> m_audioBuffer;
    std::mutex m_audioMutex;

    std::queue<std::vector<int16_t>> m_chunkQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
};