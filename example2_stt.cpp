#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <cstdlib>

#include "stt_whisper_v2.hpp"
#include "stt_config.hpp"
#include "term_util.h"
#include "audio_utils.hpp"

std::mutex g_consoleMutex;

void onTranscriptionComplete(const std::string &text, std::chrono::steady_clock::time_point capturedAt)
{
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - capturedAt)
                       .count();

    std::lock_guard<std::mutex> lock(g_consoleMutex);
    std::cout << "\n✨ [TRANSCRIPTION EVENT]: " << text
              << "  (latency ~" << latency << " ms)" << std::endl;
    std::cout << "=========================================================\n"
              << std::endl;
}

int main(int argc, char *argv[])
{
    // Resolve model path: CLI arg > STT_MODEL env var > default.
    SttConfig cfg;
    if (argc > 1)
    {
        cfg.modelPath = expand_home(argv[1]);
    }
    else if (const char *env = std::getenv("STT_MODEL"))
    {
        cfg.modelPath = expand_home(env);
    }
    else
    {
        cfg.modelPath = expand_home("~/models/ggml-base.en.bin");
    }

    TerminalInput term; // RAII: restores terminal on any exit path
    std::cout << "Initializing Continuous Voice-Activated STT Engine..." << std::endl;

    std::unique_ptr<SpeechToTextV2> sttService;
    try
    {
        sttService = std::make_unique<SpeechToTextV2>(cfg, onTranscriptionComplete);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal Initialization Error: " << e.what() << std::endl;
        return 1;
    }

    // Give the Metal shaders up to 15 seconds to warm up during model load.
    if (!sttService->startThread(cBaseWorker_V2::duration_type{15000}))
    {
        std::cerr << "Fatal Error: Failed to start the background worker thread." << std::endl;
        return 1;
    }

    term.enableRawMode();

    {
        std::lock_guard<std::mutex> lock(g_consoleMutex);
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "🎙️  Continuous Hands-Free Mode Active." << std::endl;
        std::cout << "Just start talking naturally. Pausing will auto-transcribe." << std::endl;
        std::cout << "Model: " << cfg.modelPath << std::endl;
        std::cout << "Press [ Q ] to safely shut down." << std::endl;
        std::cout << "==========================================================\n"
                  << std::endl;
    }

    bool app_running = true;
    bool sawBanner = false;

    while (app_running)
    {
        int key = term.checkKey();
        if (key == 'q' || key == 'Q')
        {
            app_running = false;
        }

        if (sttService->getState() == cBaseWorker_V2::enm_State::Stopped)
        {
            if (!sawBanner)
            {
                std::lock_guard<std::mutex> lock(g_consoleMutex);
                std::cerr << "\n⚠️ Worker pipeline stopped unexpectedly!" << std::endl;
            }
            sawBanner = true;
            app_running = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::lock_guard<std::mutex> lock(g_consoleMutex);
    std::cout << "\nShutting down continuous listeners cleanly..." << std::endl;

    static_cast<void>(sttService->stopThread());
    std::cout << "Application exited successfully." << std::endl;
    return 0;
}