#pragma once

#include "cBaseWorker_V2.h"

#include "audio_capture.hpp"
#include "stt_config.hpp"
#include "vad.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

class WhisperEngine;

// Continuous, hands-free, voice-activated speech-to-text worker (VAD based).
// Audio capture runs on the RtAudio callback thread; conversion/resampling and
// whisper inference run on this class's background worker thread.
class SpeechToTextV2 : public cBaseWorker_V2
{
public:
    // `capturedAt` is the moment the phrase began (not when transcription
    // completed), so callers can correlate with playback.
    using TranscriptionCallback =
        std::function<void(const std::string &text, std::chrono::steady_clock::time_point capturedAt)>;

    SpeechToTextV2(const SttConfig &cfg, TranscriptionCallback callback = nullptr);
    ~SpeechToTextV2() noexcept override;

    void set_callback(TranscriptionCallback callback);

protected:
    bool preRun() override;
    void run() override;
    void stopTriggered() override;

private:
    void onAudioSamples(const int16_t *samples, unsigned frames);
    void enqueuePhrase(Vad::Phrase &&phrase);

    SttConfig m_cfg;

    mutable std::mutex m_cbMutex;
    TranscriptionCallback m_callback;

    std::unique_ptr<WhisperEngine> m_engine;
    std::unique_ptr<AudioCapture> m_capture;
    std::unique_ptr<Vad> m_vad;

    struct RawPhrase
    {
        std::vector<int16_t> samples;
        std::chrono::steady_clock::time_point capturedAt{};
    };
    std::queue<RawPhrase> m_phraseQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
    size_t m_dropped = 0;
};