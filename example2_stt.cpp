#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include "stt_whisper_v2.hpp"
#include "term_util.h"

std::mutex g_consoleMutex;

void onTranscriptionComplete(const std::string &text)
{
    // Trim whitespaces
    std::string trimmed = text;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
    trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);

    // Filter out Whisper silent artifacts and hallucinations
    if (trimmed.empty() || trimmed == "[BLANK_AUDIO]" || trimmed == ".")
    {
        return;
    }

    std::lock_guard<std::mutex> lock(g_consoleMutex);
    std::cout << "\n✨ [TRANSCRIPTION EVENT]: " << trimmed << std::endl;
    std::cout << "=========================================================\n"
              << std::endl;
}

int main()
{
    TerminalInput term;
    std::string model_path = "/Users/phelelanicwele/models/ggml-base.en.bin";

    std::cout << "Initializing Continuous Voice-Activated STT Engine..." << std::endl;
    std::unique_ptr<SpeechToText> sttService;

    try
    {
        sttService = std::make_unique<SpeechToText>(model_path, onTranscriptionComplete);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal Initialization Error: " << e.what() << std::endl;
        return 1;
    }

    // Give your M1 GPU up to 15 seconds to parse and warm up the embedded metal shaders
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
        std::cout << "Press the [ Q ] key to safely shut down the application." << std::endl;
        std::cout << "==========================================================\n"
                  << std::endl;
    }

    bool app_running = true;

    while (app_running)
    {
        int key = term.checkKey();

        // Keyboard checking only tracks application shutdown states
        if (key == 'q' || key == 'Q')
        {
            app_running = false;
        }

        if (sttService->getState() == cBaseWorker_V2::enm_State::Stopped)
        {
            std::lock_guard<std::mutex> lock(g_consoleMutex);
            std::cerr << "\n⚠️ Worker pipeline stopped unexpectedly!" << std::endl;
            app_running = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::lock_guard<std::mutex> lock(g_consoleMutex);
    std::cout << "\nShutting down continuous listeners cleanly..." << std::endl;
    term.disableRawMode();

    static_cast<void>(sttService->stopThread());
    std::cout << "Application exited successfully." << std::endl;
    return 0;
}
