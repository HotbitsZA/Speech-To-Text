#pragma once

#include "stt_config.hpp"

#include <memory>
#include <string>
#include <vector>

// Encapsulates whisper.cpp context lifecycle and transcription. All heavy
// inference work happens in one thread; this class is not thread-safe.
class WhisperEngine
{
public:
    explicit WhisperEngine(const SttConfig &cfg);
    ~WhisperEngine() noexcept;

    WhisperEngine(const WhisperEngine &) = delete;
    WhisperEngine &operator=(const WhisperEngine &) = delete;

    bool valid() const noexcept { return m_context != nullptr; }

    // Runs whisper on a float32 mono chunk, returns filtered transcription text.
    // Returns an empty string for silence/hallucination artifacts so callers can
    // skip emitting events. Throws std::runtime_error on inference failure.
    std::string transcribe(const std::vector<float> &samples);

private:
    struct whisper_context *m_context = nullptr;
    SttConfig m_cfg;
};