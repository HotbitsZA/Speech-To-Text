#pragma once

#include "stt_config.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

// Pure-logic voice activity detector. No I/O; driven by feed() calls from an
// audio producer. Keeps an internal raw-PCM buffer and emits a completed Phrase
// (with its on-set timestamp) when a silence window or max-duration bound is hit.
class Vad
{
public:
    struct Phrase
    {
        std::vector<int16_t> samples;
        std::chrono::steady_clock::time_point capturedAt{};
    };

    Vad(unsigned sampleRate, const SttConfig &cfg);

    void feed(const int16_t *samples, size_t frames,
              std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

    // Returns the completed phrase (if any) and resets internal phrase state.
    // Must be drained by the same thread that calls feed().
    std::optional<Phrase> takeCompleted();

    bool calibrating() const noexcept { return m_calibrating; }
    float threshold() const noexcept { return m_threshold; }
    void requestReCalibration() noexcept { m_calibrating = false; }

private:
    void beginPhrase(std::chrono::steady_clock::time_point now);
    void finishPhrase();
    void finishCalibration();
    void appendPreroll(const int16_t *samples, size_t frames);
    size_t framesPerSecond() const noexcept { return m_sampleRate; }

    unsigned m_sampleRate;
    SttConfig m_cfg;

    // Calibration state
    bool m_calibrating = true;
    size_t m_calibrationSamples = 0;
    size_t m_targetCalibrationSamples = 0;
    std::vector<float> m_calibrationEnergies;
    float m_threshold = 0.0f;

    // Phrase state
    bool m_speaking = false;
    size_t m_silentSamples = 0;
    size_t m_silenceTimeoutSamples = 0;
    size_t m_maxPhraseSamples = 0;
    std::vector<int16_t> m_current;
    std::chrono::steady_clock::time_point m_phraseStart{};

    // Pre-roll ring (keeps audio just before speech onset)
    std::vector<int16_t> m_preroll;
    size_t m_maxPrerollSamples = 0;

    std::optional<Phrase> m_completed;
};