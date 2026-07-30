#pragma once
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>

class TerminalInput
{
private:
    struct termios original_settings;

public:
    void enableRawMode()
    {
        tcgetattr(STDIN_FILENO, &original_settings);
        struct termios raw = original_settings;
        raw.c_lflag &= ~(ECHO | ICANON); // Turn off automatic echo and line buffering
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

        // Make stdin non-blocking
        int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    }

    void disableRawMode()
    {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_settings);
    }

    // Returns the character code if pressed, or -1 if no key is pending
    int checkKey()
    {
        char ch;
        int nread = read(STDIN_FILENO, &ch, 1);
        if (nread == 1)
            return ch;
        return -1;
    }
};
