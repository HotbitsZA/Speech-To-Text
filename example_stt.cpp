#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <cstdlib>

#include "stt_whisper.hpp"
#include "stt_config.hpp"
#include "term_util.h"
#include "audio_utils.hpp"

std::mutex g_consoleMutex;

void onTranscriptionComplete(const std::string &text)
{
    std::lock_guard<std::mutex> lock(g_consoleMutex);
    std::cout << "\n🔔 [EVENT RECEIVED]: " << text << std::endl;
    std::cout << "---------------------------------------------------------\n"
              << std::endl;
}

int main(int argc, char *argv[])
{
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
    std::cout << "Initializing components..." << std::endl;

    std::unique_ptr<SpeechToTextV1> sttService;
    try
    {
        sttService = std::make_unique<SpeechToTextV1>(cfg, onTranscriptionComplete);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal Initialization Error: " << e.what() << std::endl;
        return 1;
    }

    if (!sttService->startThread(cBaseWorker_V2::duration_type{15000}))
    {
        std::cerr << "Fatal Error: Failed to start the SpeechToText background thread." << std::endl;
        return 1;
    }

    term.enableRawMode();

    {
        std::lock_guard<std::mutex> lock(g_consoleMutex);
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "🚀 Speech Engine Online & Background Worker Running." << std::endl;
        std::cout << "Hold [ R ] to stream live microphone audio." << std::endl;
        std::cout << "Release [ R ] to queue data into the background pipeline." << std::endl;
        std::cout << "Press [ Q ] to safely exit." << std::endl;
        std::cout << "==========================================================\n"
                  << std::endl;
    }

    bool app_running = true;
    const char target_key = 'r';
    bool local_recording_state = false;

    while (app_running)
    {
        int key = term.checkKey();

        if (key == 'q' || key == 'Q')
        {
            app_running = false;
        }
        else if (key == target_key)
        {
            if (!local_recording_state)
            {
                sttService->start_recording();
                local_recording_state = true;

                std::lock_guard<std::mutex> lock(g_consoleMutex);
                std::cout << "🎙️  Recording audio... (keep holding 'r')" << std::flush;
            }
        }
        else if (local_recording_state)
        {
            // Debounce to make sure the key wasn't dropped momentarily
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            if (term.checkKey() != target_key)
            {
                local_recording_state = false;
                {
                    std::lock_guard<std::mutex> lock(g_consoleMutex);
                    std::cout << " 🛑 Stopped. Audio queued for transcription." << std::endl;
                }
                sttService->stop_and_queue();
            }
        }

        if (sttService->getState() == cBaseWorker_V2::enm_State::Stopped)
        {
            std::lock_guard<std::mutex> lock(g_consoleMutex);
            std::cerr << "\n⚠️ Warning: Background worker pipeline stopped unexpectedly!" << std::endl;
            app_running = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }

    {
        std::lock_guard<std::mutex> lock(g_consoleMutex);
        std::cout << "\nShutting down system layers cleanly..." << std::endl;
    }

    term.disableRawMode();
    static_cast<void>(sttService->stopThread());

    std::cout << "Application exited successfully." << std::endl;
    return 0;
}