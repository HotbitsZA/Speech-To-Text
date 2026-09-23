#include "vad.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace
{
    constexpr float kSamplesPerSecondDiv = 1.0f / 32768.0f;
}

Vad::Vad(unsigned sampleRate, const SttConfig &cfg)
    : m_sampleRate(sampleRate),
      m_cfg(cfg),
      m_targetCalibrationSamples(static_cast<size_t>(
          static_cast<double>(m_sampleRate) * cfg.calibrationSeconds)),
      m_silenceTimeoutSamples(static_cast<size_t>(
          static_cast<double>(m_sampleRate) * cfg.pauseSeconds)),
      m_maxPhraseSamples(static_cast<size_t>(
          static_cast<double>(m_sampleRate) * cfg.maxPhraseSeconds)),
      m_maxPrerollSamples(static_cast<size_t>(
          static_cast<double>(m_sampleRate) * cfg.prerollSeconds))
{
    m_calibrationEnergies.reserve(m_targetCalibrationSamples + 4);
}

void Vad::feed(const int16_t *samples, size_t frames,
               std::chrono::steady_clock::time_point now)
{
    if (frames == 0)
    {
        return;
    }

    // Block energy = mean |x| / 32768.
    double energySum = 0.0;
    for (size_t i = 0; i < frames; ++i)
    {
        energySum += std::abs(static_cast<double>(samples[i]) * kSamplesPerSecondDiv);
    }
    const float blockEnergy = static_cast<float>(energySum / frames);

    // Calibration: sample the ambient floor, then derive a percentile-based
    // threshold instead of a simple mean (robust to transient loud spikes).
    if (m_calibrating)
    {
        m_calibrationEnergies.push_back(blockEnergy);
        m_calibrationSamples += frames;
        appendPreroll(samples, frames);
        if (m_calibrationSamples >= m_targetCalibrationSamples)
        {
            finishCalibration();
        }
        return;
    }

    const float hangoverThreshold = m_threshold * m_cfg.hysteresisRatio;

    if (!m_speaking)
    {
        if (blockEnergy > m_threshold)
        {
            beginPhrase(now);
            m_current.insert(m_current.end(), m_preroll.begin(), m_preroll.end());
            m_preroll.clear();
            m_current.insert(m_current.end(), samples, samples + frames);
        }
        else
        {
            appendPreroll(samples, frames);
        }
        return;
    }

    // Speaking: keep the trailing silence so the last word is not truncated.
    m_current.insert(m_current.end(), samples, samples + frames);

    if (blockEnergy > hangoverThreshold)
    {
        m_silentSamples = 0;
    }
    else
    {
        m_silentSamples += frames;
        if (m_silentSamples >= m_silenceTimeoutSamples)
        {
            finishPhrase();
            return;
        }
    }

    if (m_current.size() >= m_maxPhraseSamples)
    {
        finishPhrase();
    }
}

std::optional<Vad::Phrase> Vad::takeCompleted()
{
    std::optional<Phrase> p = std::move(m_completed);
    m_completed.reset();
    return p;
}

void Vad::beginPhrase(std::chrono::steady_clock::time_point now)
{
    m_speaking = true;
    m_silentSamples = 0;
    m_current.clear();
    m_phraseStart = now;
}

void Vad::finishPhrase()
{
    m_speaking = false;
    m_silentSamples = 0;

    if (!m_current.empty())
    {
        m_completed = Phrase{std::move(m_current), m_phraseStart};
        m_current.clear();
    }
}

void Vad::finishCalibration()
{
    m_calibrating = false;

    float ambient = 0.0f;
    if (!m_calibrationEnergies.empty())
    {
        // 90th percentile of the block-energy history => robust floor estimate.
        std::vector<float> sorted = m_calibrationEnergies;
        std::sort(sorted.begin(), sorted.end());
        const size_t idx = static_cast<size_t>(static_cast<double>(sorted.size() - 1) * 0.90);
        ambient = sorted[idx];
    }

    m_threshold = std::max(m_cfg.thresholdFloor, ambient * m_cfg.thresholdMultiplier);

    // Drop the raw energy history; only the derived threshold is kept.
    std::vector<float>().swap(m_calibrationEnergies);

    std::cout << "\n[VAD Calibration Complete] Ambient noise: " << ambient
              << " -> Dynamic Threshold auto-set to: " << m_threshold << "\n"
              << std::endl;
}

void Vad::appendPreroll(const int16_t *samples, size_t frames)
{
    if (m_maxPrerollSamples == 0 || frames == 0)
    {
        return;
    }
    m_preroll.insert(m_preroll.end(), samples, samples + frames);
    if (m_preroll.size() > m_maxPrerollSamples)
    {
        const size_t excess = m_preroll.size() - m_maxPrerollSamples;
        m_preroll.erase(m_preroll.begin(), m_preroll.begin() + static_cast<std::ptrdiff_t>(excess));
    }
}