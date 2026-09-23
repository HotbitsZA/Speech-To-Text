#pragma once

#include "stt_config.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// Thin RAII wrapper around RtAudio microphone capture. Owns the stream and
// delivers raw int16 mono PCM blocks to the supplied callback.
class RtAudio;

class AudioCapture
{
public:
    using SamplesCallback = std::function<void(const int16_t *samples, unsigned frames)>;

    explicit AudioCapture(const SttConfig &cfg);
    ~AudioCapture() noexcept;

    AudioCapture(const AudioCapture &) = delete;
    AudioCapture &operator=(const AudioCapture &) = delete;

    // Opens the default input device and returns its preferred sample rate.
    // Does not start the stream yet. Returns false with an error string on failure.
    bool openDefault(SamplesCallback callback, unsigned &outSampleRate, std::string &error);
    bool start(std::string &error);
    void stop() noexcept;
    void close() noexcept;

    bool streamOpen() const noexcept;
    unsigned sampleRate() const noexcept { return m_sampleRate; }

private:
    static int trampoline(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
                          double streamTime, unsigned int status, void *userData);

    SttConfig m_cfg;
    std::unique_ptr<RtAudio> m_rt;
    SamplesCallback m_callback;
    unsigned m_sampleRate = 0;
};