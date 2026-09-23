#include "stt_whisper.hpp"

#include "audio_utils.hpp"
#include "whisper_engine.hpp"

#include <iostream>
#include <string>

namespace
{
    constexpr unsigned int TARGET_WHISPER_FREQ = 16000;
}

SpeechToTextV1::SpeechToTextV1(const SttConfig &cfg, TranscriptionCallback callback)
    : cBaseWorker_V2("SpeechToTextV1"),
      m_cfg(cfg),
      m_callback(std::move(callback)),
      m_engine(std::make_unique<WhisperEngine>(cfg))
{
}

SpeechToTextV1::~SpeechToTextV1() noexcept
{
    stopThreadAndJoin();
}

void SpeechToTextV1::set_callback(TranscriptionCallback callback)
{
    std::lock_guard<std::mutex> lock(m_cbMutex);
    m_callback = std::move(callback);
}

bool SpeechToTextV1::preRun()
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

    std::cout << "[" << name() << "] Audio hardware open at " << sampleRate << " Hz" << std::endl;
    return true;
}

void SpeechToTextV1::run()
{
    while (continueRunning())
    {
        std::vector<int16_t> chunk;

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCV.wait(lock, [this]()
                           { return !m_chunkQueue.empty() || stopRequested(); });

            if (stopRequested() && m_chunkQueue.empty())
            {
                break;
            }
            if (!m_chunkQueue.empty())
            {
                chunk = std::move(m_chunkQueue.front());
                m_chunkQueue.pop();
            }
        }

        if (chunk.empty())
        {
            continue;
        }

        updateHeartbeat();

        std::vector<float> pcm = pcm16_to_float(chunk);
        if (m_capture)
        {
            pcm = resample_linear(pcm, m_capture->sampleRate(), TARGET_WHISPER_FREQ);
        }

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
            cb(text);
        }
    }
}

void SpeechToTextV1::stopTriggered()
{
    m_queueCV.notify_all();
}

void SpeechToTextV1::start_recording()
{
    if (m_recording.load(std::memory_order_acquire))
    {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(m_audioMutex);
        m_audioBuffer.clear();
    }
    m_recording.store(true, std::memory_order_release);

    if (m_capture)
    {
        std::string error;
        if (!m_capture->start(error))
        {
            std::cerr << "[" << name() << "] Error starting stream: " << error << std::endl;
        }
    }
}

void SpeechToTextV1::stop_and_queue()
{
    if (!m_recording.load(std::memory_order_acquire))
    {
        return;
    }

    m_recording.store(false, std::memory_order_release);
    if (m_capture)
    {
        m_capture->stop();
    }

    std::vector<int16_t> captured;
    {
        std::lock_guard<std::mutex> lock(m_audioMutex);
        captured = std::move(m_audioBuffer);
    }

    if (captured.empty())
    {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_chunkQueue.push(std::move(captured));
    }
    m_queueCV.notify_one();
}

void SpeechToTextV1::onAudioSamples(const int16_t *samples, unsigned frames)
{
    if (!m_recording.load(std::memory_order_relaxed))
    {
        return;
    }
    std::lock_guard<std::mutex> lock(m_audioMutex);
    m_audioBuffer.insert(m_audioBuffer.end(), samples, samples + frames);
}