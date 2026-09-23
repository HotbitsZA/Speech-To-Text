#include "whisper_engine.hpp"

#include "audio_utils.hpp"

#include <whisper.h>
#include <ggml-backend.h>

#include <limits>
#include <stdexcept>

WhisperEngine::WhisperEngine(const SttConfig &cfg)
    : m_cfg(cfg)
{
    ggml_backend_load_all();

    whisper_context_params cparams = whisper_context_default_params();
    m_context = whisper_init_from_file_with_params(m_cfg.modelPath.c_str(), cparams);
    if (!m_context)
    {
        throw std::runtime_error("WhisperEngine: failed to load model at " + m_cfg.modelPath);
    }
}

WhisperEngine::~WhisperEngine() noexcept
{
    if (m_context)
    {
        whisper_free(m_context);
        m_context = nullptr;
    }
}

std::string WhisperEngine::transcribe(const std::vector<float> &samples)
{
    if (!m_context || samples.empty())
    {
        return std::string();
    }
    if (samples.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        return std::string();
    }

    whisper_full_params wparams = whisper_full_default_params(
        m_cfg.beamSize > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_timestamps = false;
    wparams.language = m_cfg.language.c_str();
    wparams.n_threads = m_cfg.cpuThreads;
    if (m_cfg.beamSize > 1)
    {
        wparams.beam_search.beam_size = m_cfg.beamSize;
        wparams.entropy_thold = m_cfg.entropyThreshold;
    }

    if (whisper_full(m_context, wparams, samples.data(), static_cast<int>(samples.size())) != 0)
    {
        throw std::runtime_error("WhisperEngine: whisper_full failed");
    }

    std::string text;
    const int segments = whisper_full_n_segments(m_context);
    for (int i = 0; i < segments; ++i)
    {
        text += whisper_full_get_segment_text(m_context, i);
    }

    text = trim_ascii(text);
    return is_meaningful_transcription(text) ? text : std::string();
}