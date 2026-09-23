#pragma once
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>

class TerminalInput
{
private:
    struct termios original_settings{};
    bool raw_enabled = false;

public:
    ~TerminalInput()
    {
        disableRawMode();
    }

    void enableRawMode()
    {
        if (raw_enabled)
        {
            return;
        }

        if (tcgetattr(STDIN_FILENO, &original_settings) != 0)
        {
            return;
        }
        struct termios raw = original_settings;
        raw.c_lflag &= ~(ECHO | ICANON); // Turn off echo and line buffering
        if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0)
        {
            return;
        }

        // Make stdin non-blocking
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (flags != -1)
        {
            fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
        }

        raw_enabled = true;
    }

    void disableRawMode()
    {
        if (!raw_enabled)
        {
            return;
        }

        tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_settings);

        // Restore blocking behaviour
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (flags != -1)
        {
            fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
        }

        raw_enabled = false;
    }

    // Returns the character code if pressed, or -1 if no key is pending
    int checkKey()
    {
        char ch;
        int nread = read(STDIN_FILENO, &ch, 1);
        if (nread == 1)
        {
            return static_cast<unsigned char>(ch);
        }
        return -1;
    }
};