#pragma once

#include <cstddef>
#include <string>

// Central configuration for both speech-to-text compositions.
struct SttConfig
{
    std::string modelPath;

    std::string language = "en";
    int cpuThreads = 4;

    // 1 = greedy decoding, >1 = beam search.
    int beamSize = 5;
    float entropyThreshold = 2.4f;

    // VAD timing windows (seconds).
    float calibrationSeconds = 1.5f;
    float pauseSeconds = 1.2f;
    float prerollSeconds = 0.5f;
    float maxPhraseSeconds = 60.0f;

    // VAD energy tuning.
    float thresholdMultiplier = 2.5f;
    float hysteresisRatio = 0.7f;
    float thresholdFloor = 0.003f;

    // Worker queue back-pressure.
    size_t maxQueueDepth = 32;
    unsigned bufferFrames = 512;
};