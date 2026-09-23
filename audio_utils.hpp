#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

// Convert int16 PCM to normalized float32 in [-1.0, 1.0].
inline std::vector<float> pcm16_to_float(const std::vector<int16_t> &samples)
{
    std::vector<float> out(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
    {
        out[i] = static_cast<float>(samples[i]) / 32768.0f;
    }
    return out;
}

// Linear-interpolation resampler. Identity (copy) when rates match or input is empty.
inline std::vector<float> resample_linear(const std::vector<float> &in,
                                          unsigned srcRate, unsigned dstRate)
{
    if (in.empty() || srcRate == 0 || srcRate == dstRate)
    {
        return in;
    }

    const double ratio = static_cast<double>(dstRate) / static_cast<double>(srcRate);
    const size_t targetSize = in.size() <= 1
                                  ? in.size()
                                  : static_cast<size_t>(static_cast<double>(in.size() - 1) * ratio) + 1;

    std::vector<float> out(targetSize);
    for (size_t i = 0; i < targetSize; ++i)
    {
        const double srcIdx = static_cast<double>(i) / ratio;
        const size_t idxLower = static_cast<size_t>(std::floor(srcIdx));
        const size_t idxUpper = std::min(idxLower + 1, in.size() - 1);
        const double weight = srcIdx - idxLower;
        out[i] = static_cast<float>((1.0 - weight) * in[idxLower] + weight * in[idxUpper]);
    }
    return out;
}

// Trim ASCII whitespace from both ends.
inline std::string trim_ascii(const std::string &s)
{
    const auto isSpace = [](char c)
    { return std::isspace(static_cast<unsigned char>(c)); };
    size_t begin = 0;
    while (begin < s.size() && isSpace(s[begin]))
    {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin && isSpace(s[end - 1]))
    {
        --end;
    }
    return s.substr(begin, end - begin);
}

// Whisper emits silent-token artifacts under silence/noise; filter them out centrally.
inline bool is_meaningful_transcription(const std::string &text)
{
    const std::string t = trim_ascii(text);
    if (t.empty() || t == "[BLANK_AUDIO]" || t == ".")
    {
        return false;
    }
    return true;
}

// Expand a leading '~' to $HOME.
inline std::string expand_home(const std::string &path)
{
    if (path == "~")
    {
        const char *home = std::getenv("HOME");
        return home ? std::string(home) : path;
    }
    if (path.size() > 1 && path[0] == '~' && path[1] == '/')
    {
        const char *home = std::getenv("HOME");
        if (home)
        {
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}