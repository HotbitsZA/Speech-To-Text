cmake_minimum_required(VERSION 3.16)
project(WhisperSTT CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Build options
option(BUILD_LEGACY_V1 "Build the legacy push-to-talk v1 example" OFF)
option(BUILD_TESTING "Build the unit tests" ON)

# ---------------------------------------------------------------------------
# Dependencies: RtAudio via pkg-config; Whisper + GGML via pkg-config with
# find_library fallback (no hard-coded Homebrew Cellar versions).
# ---------------------------------------------------------------------------
pkg_check_modules(RTAUDIO REQUIRED IMPORTED_TARGET rtaudio)

pkg_check_modules(WHISPER QUIET IMPORTED_TARGET whisper)
if(NOT WHISPER_FOUND)
  find_library(WHISPER_LIB NAMES whisper HINTS /opt/homebrew/lib /usr/local/lib)
endif()

pkg_check_modules(GGML QUIET IMPORTED_TARGET ggml)
if(NOT GGML_FOUND)
  find_library(GGML_LIB NAMES ggml ggml-base HINTS /opt/homebrew/lib /usr/local/lib)
  if(NOT GGML_LIB)
    # Last-resort: resolve the active Homebrew install instead of pinning a version
    execute_process(COMMAND brew --prefix ggml
                    OUTPUT_VARIABLE GGML_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    find_library(GGML_LIB NAMES ggml ggml-base HINTS ${GGML_PREFIX}/lib)
  endif()
endif()

if(NOT WHISPER_FOUND AND NOT WHISPER_LIB)
  message(FATAL_ERROR "whisper was not found. Install via: brew install whisper-cpp")
endif()
if(NOT GGML_FOUND AND NOT GGML_LIB)
  message(FATAL_ERROR "ggml was not found. Install via: brew install ggml")
endif()

find_library(ACCELERATE_FRAMEWORK Accelerate)
find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
if(NOT ACCELERATE_FRAMEWORK OR NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK)
  message(FATAL_ERROR "Required Apple framework not found (Accelerate/Metal/Foundation).")
endif()

# ---------------------------------------------------------------------------
# Shared static core: WhisperEngine + AudioCapture
# ---------------------------------------------------------------------------
add_library(stt_core STATIC
    whisper_engine.cpp
    audio_capture.cpp
    vad.cpp
)
target_include_directories(stt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ThreadComponent
    /opt/homebrew/include
    /usr/local/include
)

set(STT_CORE_LINK
    PkgConfig::RTAUDIO
    Threads::Threads
    ${ACCELERATE_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)
if(WHISPER_FOUND)
  list(APPEND STT_CORE_LINK PkgConfig::WHISPER)
else()
  list(APPEND STT_CORE_LINK ${WHISPER_LIB})
endif()
if(GGML_FOUND)
  list(APPEND STT_CORE_LINK PkgConfig::GGML)
else()
  list(APPEND STT_CORE_LINK ${GGML_LIB})
endif()

target_link_libraries(stt_core PUBLIC ${STT_CORE_LINK})

# ---------------------------------------------------------------------------
# Continuous hands-free VAD app (default)
# ---------------------------------------------------------------------------
add_executable(WhisperSTTWorkerV2
    example2_stt.cpp
    stt_whisper_v2.cpp
)
target_link_libraries(WhisperSTTWorkerV2 PRIVATE stt_core)

# ---------------------------------------------------------------------------
# Optional legacy push-to-talk app
# ---------------------------------------------------------------------------
if(BUILD_LEGACY_V1)
  add_executable(example_stt
      example_stt.cpp
      stt_whisper.cpp
  )
  target_link_libraries(example_stt PRIVATE stt_core)
  set_target_properties(example_stt PROPERTIES OUTPUT_NAME "example_stt.bin")
endif()

# ---------------------------------------------------------------------------
# Unit tests (pure logic: resampler + VAD, no audio hardware needed)
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  enable_testing()
  add_executable(stt_tests tests/test_audio_utils.cpp)
  target_link_libraries(stt_tests PRIVATE stt_core)
  add_test(NAME stt_tests COMMAND stt_tests)
endif()

foreach(target WhisperSTTWorkerV2 stt_core example_stt stt_tests)
  if(TARGET ${target})
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endforeach()