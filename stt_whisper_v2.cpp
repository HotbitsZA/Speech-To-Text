#include "stt_whisper_v2.hpp"
#include <whisper.h>
#include <ggml-backend.h>
#include <RtAudio.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <numeric>

namespace
{
    constexpr unsigned int TARGET_WHISPER_FREQ = 16000;
    constexpr unsigned int BUFFER_FRAMES = 512;
    constexpr float PAUSE_DURATION_SECONDS = 1.2f;   // Time of silence to trigger a text segment split
    constexpr float PREROLL_DURATION_SECONDS = 0.5f; // Keeps audio frames right before you start talking

    bool file_exists(const std::string &path)
    {
        return std::ifstream(path).good();
    }
}

class SpeechToText::Impl
{
public:
    unsigned int nativeSampleRate = TARGET_WHISPER_FREQ;
    std::vector<int16_t> prerollBuffer;
    size_t maxPrerollSamples = 0;

    // Automatic noise calibration variables
    bool isCalibrating = true;
    size_t calibrationSamplesAccumulated = 0;
    double calibrationEnergySum = 0.0;
    size_t targetCalibrationSamples = 0;
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
    if (!self || !inputBuffer)
        return 0;

    const int16_t *in = static_cast<const int16_t *>(inputBuffer);
    self->process_incoming_samples(in, nBufferFrames);
    return 0;
}

void SpeechToText::process_incoming_samples(const int16_t *samples, unsigned int frames)
{
    std::lock_guard<std::mutex> lock(audio_mutex);

    // 1. Calculate the Absolute Average Amplitude energy of this block
    double blockEnergySum = 0.0;
    for (unsigned int i = 0; i < frames; ++i)
    {
        blockEnergySum += std::abs(static_cast<double>(samples[i]) / 32768.0);
    }
    float currentEnergy = static_cast<float>(blockEnergySum / frames);

    // 2. Handle Automatic Noise Floor Calibration during the first 1.5 seconds
    if (m_pImpl->isCalibrating)
    {
        m_pImpl->calibrationEnergySum += blockEnergySum;
        m_pImpl->calibrationSamplesAccumulated += frames;

        if (m_pImpl->calibrationSamplesAccumulated >= m_pImpl->targetCalibrationSamples)
        {
            // calibrationEnergySum is the sum of |x|/32768 across every captured sample,
            // so dividing by the total sample count yields the true per-sample average.
            // (The previous code divided by the block frame count a second time, reporting
            // an ambient floor ~512x lower than reality and silently pinning the threshold
            // to its 0.003 clamp.)
            float averageAmbientNoise = static_cast<float>(m_pImpl->calibrationEnergySum / m_pImpl->calibrationSamplesAccumulated);

            // Set VAD threshold dynamically to 2.5x the ambient background noise floor
            m_vadThreshold = std::max(0.003f, averageAmbientNoise * 2.5f);
            m_pImpl->isCalibrating = false;

            std::cout << "\n[VAD Calibration Complete] Ambient noise: " << averageAmbientNoise
                      << " -> Dynamic Threshold auto-set to: " << m_vadThreshold << "\n"
                      << std::endl;
        }
        return; // Skip speech tracking while calibrating
    }

    // 3. Process Live Voice Activity Detection
    if (currentEnergy > m_vadThreshold)
    {
        // User is actively speaking
        if (!m_isSpeaking)
        {
            m_isSpeaking = true;
            m_phraseCaptureStart = std::chrono::steady_clock::now();
            // Prepend our saved pre-roll audio history so the start of the first word isn't cut off
            audio_buffer.insert(audio_buffer.end(), m_pImpl->prerollBuffer.begin(), m_pImpl->prerollBuffer.end());
            m_pImpl->prerollBuffer.clear();
        }

        audio_buffer.insert(audio_buffer.end(), samples, samples + frames);
        m_consecutiveSilenceSamples = 0;
    }
    else
    {
        // User is silent
        if (m_isSpeaking)
        {
            // Keep collecting frames during the pause window to prevent truncating trailing words
            audio_buffer.insert(audio_buffer.end(), samples, samples + frames);
            m_consecutiveSilenceSamples += frames;

            if (m_consecutiveSilenceSamples >= m_silenceTimeoutSamples)
            {
                slice_and_queue_active_phrase();
            }
        }
        else
        {
            // Maintain a rolling history buffer for the pre-roll window
            m_pImpl->prerollBuffer.insert(m_pImpl->prerollBuffer.end(), samples, samples + frames);
            if (m_pImpl->prerollBuffer.size() > m_pImpl->maxPrerollSamples)
            {
                m_pImpl->prerollBuffer.erase(m_pImpl->prerollBuffer.begin(),
                                             m_pImpl->prerollBuffer.begin() + (m_pImpl->prerollBuffer.size() - m_pImpl->maxPrerollSamples));
            }
        }
    }
}

void SpeechToText::slice_and_queue_active_phrase()
{
    std::vector<int16_t> captured_audio = std::move(audio_buffer);
    audio_buffer.clear();
    m_isSpeaking = false;
    m_consecutiveSilenceSamples = 0;

    if (captured_audio.empty())
        return;

    // Convert raw PCM int16 array to floating-point normalized buffers
    std::vector<float> pcmf32_native(captured_audio.size());
    for (size_t i = 0; i < captured_audio.size(); ++i)
    {
        pcmf32_native[i] = static_cast<float>(captured_audio[i]) / 32768.0f;
    }

    // High-Fidelity Downsampler to 16000Hz
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
        m_taskQueue.push({std::move(pcmf32_whisper), m_phraseCaptureStart});
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

    RtAudio::DeviceInfo info = adc->getDeviceInfo(params.deviceId);
    m_pImpl->nativeSampleRate = info.preferredSampleRate;

    // Calculate sample sizes for our timing windows
    m_silenceTimeoutSamples = static_cast<size_t>(m_pImpl->nativeSampleRate * PAUSE_DURATION_SECONDS);
    m_pImpl->maxPrerollSamples = static_cast<size_t>(m_pImpl->nativeSampleRate * PREROLL_DURATION_SECONDS);
    m_pImpl->targetCalibrationSamples = static_cast<size_t>(m_pImpl->nativeSampleRate * 1.5f); // 1.5 seconds calibration window

    try
    {
        adc->openStream(nullptr, &params, RTAUDIO_SINT16, m_pImpl->nativeSampleRate, &bufferFrames, &SpeechToText::audio_callback, this);
        adc->startStream();
        std::cout << "[" << name() << "] Hands-free VAD Engine Online. Calibrating mic floor... (Please stay silent briefly)" << std::endl;
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
        st_CapturedPhrase phraseToProcess;

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCV.wait(lock, [this]()
                           { return !m_taskQueue.empty() || stopRequested(); });

            if (stopRequested() && m_taskQueue.empty())
                break;

            if (!m_taskQueue.empty())
            {
                phraseToProcess = std::move(m_taskQueue.front());
                m_taskQueue.pop();
            }
        }

        if (!phraseToProcess.samples.empty())
        {
            updateHeartbeat();

            // Beam Search configurations
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
            wparams.print_progress = false;
            wparams.print_special = false;
            wparams.print_timestamps = false;
            wparams.language = "en";
            wparams.n_threads = 4;

            wparams.beam_search.beam_size = 5;
            wparams.entropy_thold = 2.4f;

            if (whisper_full(ctx, wparams, phraseToProcess.samples.data(), static_cast<int>(phraseToProcess.samples.size())) != 0)
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
                currentCallback(text_output, phraseToProcess.capturedAt);
        }
    }
}
void SpeechToText::stopTriggered()
{
    m_queueCV.notify_all();
}
