#pragma once

#include "cBaseWorker_V2.h"
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>

struct whisper_context;
class RtAudio;

class SpeechToText : public cBaseWorker_V2
{
public:
    // `capturedAt` is the moment the learner's phrase began (not when transcription
    // completed), so callers can tell whether the audio overlapped the teacher's own
    // playback even though whisper finishes asynchronously.
    using TranscriptionCallback = std::function<void(const std::string &text, std::chrono::steady_clock::time_point capturedAt)>;

    SpeechToText(const std::string &model_path, TranscriptionCallback callback = nullptr);
    ~SpeechToText() noexcept override;

    // Dynamically change or assign the text completion event listener
    void set_callback(TranscriptionCallback callback);

protected:
    bool preRun() override;
    void run() override;
    void stopTriggered() override;

private:
    static int audio_callback(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
                              double streamTime, unsigned int status, void *userData);

    // Dynamic processing internal functions
    void process_incoming_samples(const int16_t *samples, unsigned int frames);
    void slice_and_queue_active_phrase();

    std::string m_modelPath;
    TranscriptionCallback m_onTranscriptionComplete;
    struct whisper_context *ctx = nullptr;
    std::unique_ptr<RtAudio> adc;

    // VAD & Energy State Parameters
    std::vector<int16_t> audio_buffer;
    std::mutex audio_mutex;

    // VAD Variables
    float m_vadThreshold = 0.02f;           // RMS Amplitude Threshold (Adjust based on mic sensitivity)
    size_t m_silenceTimeoutSamples = 0;     // Number of native samples representing a punctuation pause
    size_t m_consecutiveSilenceSamples = 0; // Running silence frame counter
    bool m_isSpeaking = false;              // Tracking variable for current phrasing phase
    std::chrono::steady_clock::time_point m_phraseCaptureStart{}; // Wall-clock onset of the current phrase

    // Deep Asynchronous Execution Pipelines
    struct st_CapturedPhrase
    {
        std::vector<float> samples;
        std::chrono::steady_clock::time_point capturedAt{};
    };
    std::queue<st_CapturedPhrase> m_taskQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;

    class Impl;
    std::unique_ptr<Impl> m_pImpl;
};
