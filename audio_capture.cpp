#include "audio_capture.hpp"

#include <RtAudio.h>

#include <utility>

AudioCapture::AudioCapture(const SttConfig &cfg)
    : m_cfg(cfg)
{
}

AudioCapture::~AudioCapture() noexcept
{
    try
    {
        close();
    }
    catch (...)
    {
    }
}

bool AudioCapture::openDefault(SamplesCallback callback, unsigned &outSampleRate, std::string &error)
{
    m_callback = std::move(callback);

    auto adc = std::make_unique<RtAudio>();
    if (adc->getDeviceCount() < 1)
    {
        error = "No RtAudio input devices found";
        return false;
    }

    RtAudio::StreamParameters params;
    params.deviceId = adc->getDefaultInputDevice();
    params.nChannels = 1;
    params.firstChannel = 0;

    RtAudio::DeviceInfo info = adc->getDeviceInfo(params.deviceId);
    m_sampleRate = info.preferredSampleRate;
    unsigned bufferFrames = m_cfg.bufferFrames;

    try
    {
        adc->openStream(nullptr, &params, RTAUDIO_SINT16, m_sampleRate, &bufferFrames,
                        &AudioCapture::trampoline, this);
    }
    catch (const std::exception &e)
    {
        error = e.what();
        return false;
    }

    m_rt = std::move(adc);
    outSampleRate = m_sampleRate;
    return true;
}

bool AudioCapture::start(std::string &error)
{
    if (!m_rt)
    {
        error = "AudioCapture: stream not open";
        return false;
    }
    try
    {
        m_rt->startStream();
        return true;
    }
    catch (const std::exception &e)
    {
        error = e.what();
        return false;
    }
}

void AudioCapture::stop() noexcept
{
    try
    {
        if (m_rt && m_rt->isStreamRunning())
        {
            m_rt->stopStream();
        }
    }
    catch (...)
    {
    }
}

void AudioCapture::close() noexcept
{
    try
    {
        if (m_rt && m_rt->isStreamOpen())
        {
            m_rt->closeStream();
        }
    }
    catch (...)
    {
    }
    m_rt.reset();
}

bool AudioCapture::streamOpen() const noexcept
{
    return m_rt && m_rt->isStreamOpen();
}

int AudioCapture::trampoline(void *outputBuffer, void *inputBuffer, unsigned int nBufferFrames,
                             double streamTime, unsigned int status, void *userData)
{
    (void)outputBuffer;
    (void)streamTime;
    (void)status;

    auto *self = static_cast<AudioCapture *>(userData);
    if (!self || !inputBuffer || !self->m_callback)
    {
        return 0;
    }
    self->m_callback(static_cast<const int16_t *>(inputBuffer), nBufferFrames);
    return 0;
}