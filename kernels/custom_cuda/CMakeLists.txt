cmake_minimum_required(VERSION 3.15 FATAL_ERROR) # CMake 3.15+ is good for CUDA support

# Set the policy for FindCUDA module removal to NEW.
# This helps ensure CMake uses its modern first-class CUDA language features
# and avoids the old FindCUDA module's behavior that might not define CUDA::cudart.
cmake_policy(SET CMP0146 NEW)

project(NeuralNetProject LANGUAGES CXX CUDA) # Declare both C++ and CUDA languages

# --- Essential for LSP Integration (e.g., clangd) ---
# This generates compile_commands.json in your build directory,
# which LSPs use to understand compilation flags and include paths.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17) # Or 20, 23 etc., depending on your project's needs
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard-compliant C++

include(CTest)
# --- Fetch GoogleTest ---
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    # This option addresses a warning about file timestamps.
    # It ensures that when the archive is extracted, the file timestamps are updated.
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


# --- Define Source Files for Main Executable ---
# Find all .cu and .cuh files for the main program.
# We explicitly exclude the Tests directory here.
file(GLOB_RECURSE MAIN_SOURCES
    Data/*.cu
    Layers/*.cu
    main.cu
    neuralnet.cu
)

# --- Define Source Files for Test Executable ---
# Find all .cu and .cuh files specifically for the tests.
# This has been updated to also explicitly check for `main_test.cu`
# in the current source directory, just in case it's not in the Tests folder.
file(GLOB_RECURSE TEST_SOURCES
    tests/*.cu
    main_test.cu
)

# For debugging, let's print the test sources found.
message(STATUS "Found test sources: ${TEST_SOURCES}")

# --- Define Include Directories ---
# These include paths are common for both the main and test executables.
# We'll create a list of them to apply to both targets.
set(COMMON_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Data
    ${CMAKE_CURRENT_SOURCE_DIR}/Layers
)

# --- Create Main Executable Target ---
# This executable will contain all the core application logic.
add_executable(${PROJECT_NAME} ${MAIN_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${COMMON_INCLUDE_DIRS})
# Link against the CUDA runtime library, which is essential for CUDA code.
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})

# --- Create Test Executable Target ---
# This is a separate executable specifically for running your tests.
add_executable(NeuralNetTests ${TEST_SOURCES})
target_include_directories(NeuralNetTests PUBLIC ${COMMON_INCLUDE_DIRS})

# Link the test executable to the CUDA runtime and the GoogleTest library.
# The 'GTest::gtest_main' target is deprecated and it's better to
# link with 'GTest::gtest'.
target_link_libraries(NeuralNetTests PRIVATE ${CUDA_LIBRARIES} GTest::gtest)

# --- Enable Testing ---
# This enables CTest and sets up the test discovery.
enable_testing()

# --- GoogleTest Configuration ---
# The 'GoogleTest' module must be included to use the gtest_discover_tests command.
include(GoogleTest)
gtest_discover_tests(NeuralNetTests)

# --- CUDA Specific Settings ---
# Set the target GPU architecture for all targets in the project.
set(CUDA_ARCH_CODE "86")
set_property(TARGET ${PROJECT_NAME} NeuralNetTests PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_CODE})
