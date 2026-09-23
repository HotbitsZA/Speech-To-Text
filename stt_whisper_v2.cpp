#include "stt_whisper_v2.hpp"

#include "audio_utils.hpp"
#include "whisper_engine.hpp"

#include <iostream>
#include <optional>
#include <string>

namespace
{
    constexpr unsigned int TARGET_WHISPER_FREQ = 16000;
}

SpeechToTextV2::SpeechToTextV2(const SttConfig &cfg, TranscriptionCallback callback)
    : cBaseWorker_V2("SpeechToTextV2"),
      m_cfg(cfg),
      m_callback(std::move(callback)),
      m_engine(std::make_unique<WhisperEngine>(cfg))
{
}

SpeechToTextV2::~SpeechToTextV2() noexcept
{
    // Guarantee the worker thread (which uses m_engine/m_capture) has fully
    // exited before those members are destroyed.
    stopThreadAndJoin();
}

void SpeechToTextV2::set_callback(TranscriptionCallback callback)
{
    std::lock_guard<std::mutex> lock(m_cbMutex);
    m_callback = std::move(callback);
}

bool SpeechToTextV2::preRun()
{
    if (!m_engine || !m_engine->valid())
    {
        return false;
    }

    m_capture = std::make_unique<AudioCapture>(m_cfg);

    unsigned sampleRate = 0;
    std::string error;
    if (!m_capture->openDefault(
            [this](const int16_t *samples, unsigned frames) { onAudioSamples(samples, frames); },
            sampleRate, error))
    {
        std::cerr << "[" << name() << "] Capture open error: " << error << std::endl;
        return false;
    }

    m_vad = std::make_unique<Vad>(sampleRate, m_cfg);

    if (!m_capture->start(error))
    {
        std::cerr << "[" << name() << "] Capture start error: " << error << std::endl;
        return false;
    }

    std::cout << "[" << name() << "] Hands-free VAD online at " << sampleRate
              << " Hz. Calibrating mic floor... (please stay quiet briefly)" << std::endl;
    return true;
}

void SpeechToTextV2::run()
{
    while (continueRunning())
    {
        RawPhrase phrase;

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCV.wait(lock, [this]()
                           { return !m_phraseQueue.empty() || stopRequested(); });

            if (stopRequested() && m_phraseQueue.empty())
            {
                break;
            }
            if (!m_phraseQueue.empty())
            {
                phrase = std::move(m_phraseQueue.front());
                m_phraseQueue.pop();
            }
        }

        if (phrase.samples.empty())
        {
            continue;
        }

        updateHeartbeat();

        // Conversion + resampling live on the worker thread, never on the audio thread.
        std::vector<float> pcm = pcm16_to_float(phrase.samples);
        const unsigned nativeRate = m_capture ? m_capture->sampleRate() : TARGET_WHISPER_FREQ;
        pcm = resample_linear(pcm, nativeRate, TARGET_WHISPER_FREQ);

        std::string text;
        try
        {
            text = m_engine->transcribe(pcm);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << name() << "] Whisper failure: " << e.what() << std::endl;
            continue;
        }

        if (text.empty())
        {
            continue;
        }

        TranscriptionCallback cb;
        {
            std::lock_guard<std::mutex> lock(m_cbMutex);
            cb = m_callback;
        }
        if (cb)
        {
            cb(text, phrase.capturedAt);
        }
    }
}

void SpeechToTextV2::stopTriggered()
{
    m_queueCV.notify_all();
}

void SpeechToTextV2::onAudioSamples(const int16_t *samples, unsigned frames)
{
    if (m_vad)
    {
        m_vad->feed(samples, frames);
    }

    std::optional<Vad::Phrase> phrase;
    while ((phrase = m_vad ? m_vad->takeCompleted() : std::nullopt))
    {
        enqueuePhrase(std::move(*phrase));
    }
}

void SpeechToTextV2::enqueuePhrase(Vad::Phrase &&phrase)
{
    std::lock_guard<std::mutex> lock(m_queueMutex);

    if (m_phraseQueue.size() >= m_cfg.maxQueueDepth)
    {
        // Back-pressure: keep the newest phrase, drop the oldest.
        m_phraseQueue.pop();
        ++m_dropped;
        if (m_dropped == 1 || (m_dropped % 32) == 0)
        {
            std::cerr << "[" << name() << "] WARNING: transcription backlog, dropped "
                      << m_dropped << " phrase(s)" << std::endl;
        }
    }

    m_phraseQueue.push(RawPhrase{std::move(phrase.samples), phrase.capturedAt});
    m_queueCV.notify_one();
}