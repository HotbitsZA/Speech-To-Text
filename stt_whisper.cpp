#include "stt_whisper.hpp"
#include <whisper.h>
#include <ggml-backend.h>
#include <RtAudio.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cmath>

namespace
{
    constexpr unsigned int TARGET_WHISPER_FREQ = 16000;
    constexpr unsigned int BUFFER_FRAMES = 512;

    bool file_exists(const std::string &path)
    {
        return std::ifstream(path).good();
    }
}

class SpeechToText::Impl
{
public:
    unsigned int nativeSampleRate = TARGET_WHISPER_FREQ;
};

SpeechToText::SpeechToText(const std::string &model_path, TranscriptionCallback callback)
    : cBaseWorker_V2("SpeechToTextWorker"),
      m_modelPath(model_path),
      m_onTranscriptionComplete(std::move(callback)),
      ctx(nullptr),
      m_pImpl(std::make_unique<Impl>())
{
    if (!file_exists(m_modelPath))
    {
        throw std::runtime_error("STT Error: Whisper model file not found at " + model_path);
    }
}

SpeechToText::~SpeechToText() noexcept
{
    static_cast<void>(stopThread());
    try
    {
        if (adc && adc->isStreamOpen())
        {
            adc->closeStream();
        }
    }
    catch (...)
    {
    }

    if (ctx)
    {
        whisper_free(ctx);
        ctx = nullptr;
    }
}

int SpeechToText::audio_callback(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
                                 double streamTime, unsigned int status, void *userData)
{
    (void)outputBuffer;
    (void)streamTime;
    (void)status;
    auto *self = static_cast<SpeechToText *>(userData);
    if (!self || !self->is_recording.load(std::memory_order_relaxed) || !inputBuffer)
        return 0;

    const int16_t *in = static_cast<const int16_t *>(inputBuffer);
    std::lock_guard<std::mutex> lock(self->audio_mutex);
    self->audio_buffer.insert(self->audio_buffer.end(), in, in + nBufferFrames);
    return 0;
}

void SpeechToText::start_recording()
{
    if (is_recording.load(std::memory_order_acquire))
        return;

    {
        std::lock_guard<std::mutex> lock(audio_mutex);
        audio_buffer.clear();
    }
    is_recording.store(true, std::memory_order_release);

    if (adc && adc->isStreamOpen() && !adc->isStreamRunning())
    {
        try
        {
            adc->startStream();
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << name() << "] Error starting audio stream: " << e.what() << std::endl;
        }
    }
}

void SpeechToText::stop_and_queue()
{
    if (!is_recording.load(std::memory_order_acquire))
        return;

    is_recording.store(false, std::memory_order_release);
    if (adc && adc->isStreamRunning())
    {
        try
        {
            adc->stopStream();
        }
        catch (const std::exception &e)
        {
            std::cerr << "[" << name() << "] Error stopping audio stream: " << e.what() << std::endl;
        }
    }

    std::vector<int16_t> captured_audio;
    {
        std::lock_guard<std::mutex> lock(audio_mutex);
        captured_audio = std::move(audio_buffer);
    }

    if (captured_audio.empty())
        return;

    // 1. Convert raw PCM int16 to normalized float32
    std::vector<float> pcmf32_native(captured_audio.size());
    for (size_t i = 0; i < captured_audio.size(); ++i)
    {
        pcmf32_native[i] = static_cast<float>(captured_audio[i]) / 32768.0f;
    }

    // 2. High-Fidelity Software Resampling to 16000Hz if hardware is different
    std::vector<float> pcmf32_whisper;
    if (m_pImpl->nativeSampleRate == TARGET_WHISPER_FREQ)
    {
        pcmf32_whisper = std::move(pcmf32_native);
    }
    else
    {
        double resampleRatio = static_cast<double>(TARGET_WHISPER_FREQ) / m_pImpl->nativeSampleRate;
        size_t targetSize = static_cast<size_t>(pcmf32_native.size() * resampleRatio);
        pcmf32_whisper.resize(targetSize);

        for (size_t i = 0; i < targetSize; ++i)
        {
            double srcIdx = i / resampleRatio;
            size_t idxLower = static_cast<size_t>(std::floor(srcIdx));
            size_t idxUpper = std::min(idxLower + 1, pcmf32_native.size() - 1);
            double weight = srcIdx - idxLower;

            pcmf32_whisper[i] = (1.0 - weight) * pcmf32_native[idxLower] + weight * pcmf32_native[idxUpper];
        }
    }

    if (pcmf32_whisper.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        return;

    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_taskQueue.push(std::move(pcmf32_whisper));
    }
    m_queueCV.notify_one();
}

void SpeechToText::set_callback(TranscriptionCallback callback)
{
    std::lock_guard<std::mutex> lock(m_queueMutex);
    m_onTranscriptionComplete = std::move(callback);
}

bool SpeechToText::preRun()
{
    ggml_backend_load_all();

    struct whisper_context_params cparams = whisper_context_default_params();
    ctx = whisper_init_from_file_with_params(m_modelPath.c_str(), cparams);
    if (!ctx)
        return false;

    adc = std::make_unique<RtAudio>();
    if (adc->getDeviceCount() < 1)
        return false;

    RtAudio::StreamParameters params;
    params.deviceId = adc->getDefaultInputDevice();
    params.nChannels = 1;
    params.firstChannel = 0;
    unsigned int bufferFrames = BUFFER_FRAMES;

    // Query hardware preferred format rather than forcing cross-system alterations
    RtAudio::DeviceInfo info = adc->getDeviceInfo(params.deviceId);
    m_pImpl->nativeSampleRate = info.preferredSampleRate;

    try
    {
        adc->openStream(nullptr, &params, RTAUDIO_SINT16, m_pImpl->nativeSampleRate, &bufferFrames, &SpeechToText::audio_callback, this);
        std::cout << "[" << name() << "] Audio hardware successfully opened at native " << m_pImpl->nativeSampleRate << " Hz" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[" << name() << "] STT Error opening stream: " << e.what() << std::endl;
        return false;
    }

    return true;
}

void SpeechToText::run()
{
    while (continueRunning())
    {
        std::vector<float> chunkToProcess;

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCV.wait(lock, [this]()
                           { return !m_taskQueue.empty() || stopRequested(); });

            if (stopRequested() && m_taskQueue.empty())
                break;

            if (!m_taskQueue.empty())
            {
                chunkToProcess = std::move(m_taskQueue.front());
                m_taskQueue.pop();
            }
        }

        if (!chunkToProcess.empty())
        {
            updateHeartbeat();

            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
            wparams.print_progress = false;
            wparams.print_special = false;
            wparams.print_timestamps = false;
            wparams.language = "en";
            wparams.n_threads = 4;

            if (whisper_full(ctx, wparams, chunkToProcess.data(), static_cast<int>(chunkToProcess.size())) != 0)
                continue;

            std::string text_output = "";
            int segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < segments; ++i)
            {
                text_output += whisper_full_get_segment_text(ctx, i);
            }

            TranscriptionCallback currentCallback;
            {
                std::lock_guard<std::mutex> lock(m_queueMutex);
                currentCallback = m_onTranscriptionComplete;
            }

            if (currentCallback)
                currentCallback(text_output);
        }
    }
}

void SpeechToText::stopTriggered()
{
    m_queueCV.notify_all();
}
