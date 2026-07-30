cmake_minimum_required(VERSION 3.16)
project(WhisperSTTWorkerV2 CXX)

# Force modern C++ standards required by cBaseWorker_V2
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Locate the native system multi-threading packages (REQUIRED for Threads::Threads)
find_package(Threads REQUIRED)

# 2. Use PkgConfig toolset to locate Homebrew's RtAudio installation components
find_package(PkgConfig REQUIRED)
pkg_check_modules(RTAUDIO REQUIRED IMPORTED_TARGET rtaudio)

# 3. Add Homebrew search paths to discover libwhisper.a automatically
find_library(WHISPER_LIB 
    NAMES whisper
    HINTS /opt/homebrew/lib /usr/local/lib
    REQUIRED
)

# FIX: Find the Homebrew-installed GGML base shared library containing the missing initialization symbols
find_library(GGML_LIB 
    NAMES ggml ggml-base
    HINTS /opt/homebrew/lib /usr/local/lib /opt/homebrew/Cellar/ggml/0.17.0/lib
    REQUIRED
)

# 4. Locate native Apple Frameworks for high-speed M1 GPU/CPU acceleration
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# 5. Group your project implementation source targets together
add_executable(${PROJECT_NAME} 
    example2_stt.cpp 
    stt_whisper_v2.cpp
)

# 6. Handle directory mappings (Current, ThreadComponent parent paths, and Homebrew variables)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ThreadComponent
    /opt/homebrew/include
    /opt/homebrew/Cellar/ggml/0.17.0/include
    /usr/local/include
)

# 7. Link libraries, Apple acceleration layers, and system threads
target_link_libraries(${PROJECT_NAME} PRIVATE
    PkgConfig::RTAUDIO            # Resolves the Homebrew PkgConfig target map
    ${WHISPER_LIB}
    ${GGML_LIB}                   # FIX: Links the library resolving _ggml_backend_load_all
    ${ACCELERATE_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    Threads::Threads              
)

# Enable build cleaner warning outputs
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
