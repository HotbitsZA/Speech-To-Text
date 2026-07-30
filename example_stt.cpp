#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include "stt_whisper.hpp"
#include "term_util.h"

// Synchronises terminal writes between the main UI thread and the background worker callback thread
std::mutex g_consoleMutex;

// This callback fires asynchronously on the background worker's thread
void onTranscriptionComplete(const std::string &text)
{
    // 1. Create a local copy and trim wrapping white-spaces
    std::string trimmed = text;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
    trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);

    // 2. Suppress terminal output for silent tokens, empty buffers, or hallucinations
    if (trimmed.empty() || trimmed == "[BLANK_AUDIO]" || trimmed == ".")
    {
        return;
    }

    // 3. Print out your clear, valid speech segments safely
    std::lock_guard<std::mutex> lock(g_consoleMutex);
    std::cout << "\n🔔 [EVENT RECEIVED]: New Text Arrived!" << std::endl;
    std::cout << "📝 Text: " << trimmed << std::endl;
    std::cout << "---------------------------------------------------------\n"
              << std::endl;
}

int main()
{
    TerminalInput term;

    // 1. Initialise the background worker and set up the callback listener
    std::string model_path = "/Users/phelelanicwele/models/ggml-base.en.bin";

    std::cout << "Initializing components..." << std::endl;
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

    // 2. Launch the cBaseWorker execution thread
    if (!sttService->startThread(cBaseWorker_V2::duration_type{15000}))
    {
        std::cerr << "Fatal Error: Failed to start the SpeechToText background thread." << std::endl;
        return 1;
    }

    // 3. Configure and Enter Terminal Capture Raw Mode
    term.enableRawMode();

    {
        std::lock_guard<std::mutex> lock(g_consoleMutex);
        std::cout << "\n==========================================================" << std::endl;
        std::cout << "🚀 Speech Engine Online & Background Worker Running." << std::endl;
        std::cout << "Hold down the [ R ] key to stream live microphone audio." << std::endl;
        std::cout << "Release the [ R ] key to queue data into the background pipeline." << std::endl;
        std::cout << "Press [ Q ] to safely exit the application." << std::endl;
        std::cout << "==========================================================\n"
                  << std::endl;
    }

    bool app_running = true;
    char target_key = 'r';
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
                // Instantly notify hardware pipeline to start recording inputs
                sttService->start_recording();
                local_recording_state = true;

                std::lock_guard<std::mutex> lock(g_consoleMutex);
                std::cout << "🎙️  Recording audio... (Keep holding '" << target_key << "')" << std::flush;
            }
        }
        else if (local_recording_state)
        {
            // Debounce sleep to make sure the key wasn't dropped momentarily
            std::this_thread::sleep_for(std::chrono::milliseconds(120));
            if (term.checkKey() != target_key)
            {
                local_recording_state = false;

                {
                    std::lock_guard<std::mutex> lock(g_consoleMutex);
                    std::cout << " 🛑 Stopped. Audio pushed to background thread." << std::endl;
                }

                // Extracts raw audio, normalizes it, and notifies the internal condition variables
                sttService->stop_and_queue();
            }
        }

        // Check background worker health status metrics from the parent class framework
        if (sttService->getState() == cBaseWorker_V2::enm_State::Stopped)
        {
            std::lock_guard<std::mutex> lock(g_consoleMutex);
            std::cerr << "\n⚠️ Warning: Background worker pipeline stopped unexpectedly!" << std::endl;
            app_running = false;
        }

        // Keep the main polling thread light and responsive
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }

    // 4. Safe Resource Teardown Sequence
    {
        std::lock_guard<std::mutex> lock(g_consoleMutex);
        std::cout << "\nShutting down system layers cleanly..." << std::endl;
    }

    term.disableRawMode();

    // Explicitly stops the thread. The class destructor also handles this via RAII rules.
    static_cast<void>(sttService->stopThread());

    std::cout << "Application exited successfully." << std::endl;
    return 0;
}
