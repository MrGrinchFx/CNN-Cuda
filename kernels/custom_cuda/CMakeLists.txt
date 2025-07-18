cmake_minimum_required(VERSION 3.15 FATAL_ERROR) # Require a modern CMake version

project(NeuralNetworkProject LANGUAGES CXX CUDA) # Declare project name and supported languages

# Set common compiler flags for C++
set(CMAKE_CXX_STANDARD 17) # Use C++17 standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Don't use GNU extensions by default

# Add typical warnings and optimizations for C++
add_compile_options(-Wall -Wextra -Wpedantic) # Enable common warnings
if (CMAKE_BUILD_TYPE MATCHES "Debug")
  add_compile_options(-g) # Enable debug symbols
else()
  add_compile_options(-O3 -DNDEBUG) # Enable optimizations for release, disable asserts
endif()

# Set common compiler flags for CUDA
# This uses the CUDA compiler driver (nvcc) and ensures consistency with C++ flags
set(CMAKE_CUDA_STANDARD 17) # Use C++17 standard for CUDA host code
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Add typical warnings and optimizations for CUDA
# These are passed to nvcc
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75> # Example: Compile for compute capability 7.5 (adjust for your GPU)
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets> # Suppress warnings about older GPU targets
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true> # Enable relocatable device code (required for separate compilation)
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> # Relax constexpr rules for older nvcc versions
)
if (CMAKE_BUILD_TYPE MATCHES "Debug")
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-g -G>) # Enable debug symbols and device-side debugging
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>) # Enable optimizations for CUDA
endif()

# Find CUDA toolkit
find_package(CUDA REQUIRED)

# Specify include directories
# This is where your header files (like utils.hpp) reside
include_directories(. ) # Current directory is where utils.hpp is

# --- Define Libraries (if you plan to separate compilation) ---
# It looks like layers.cu and layerKernels.cu might form a library.
# This approach promotes modularity and faster compilation for large projects.

# Add an executable target
# Assuming main.cu contains the main function for the final application.
# Combine all source files into one executable.
add_executable(neural_network_app
    main.cu
    main.cpp
    neuralnet.cpp
    layers.cu
    layerKernels.cu
)

# Link the executable with CUDA libraries (cudart for runtime API)
target_link_libraries(neural_network_app PRIVATE CUDA::cudart)

# If you had more complex dependencies or third-party libraries, you'd add them here:
# target_link_libraries(neural_network_app PRIVATE other_library_name)

# --- Optional: Install rules ---
# install(TARGETS neural_network_app DESTINATION bin)

# --- Optional: Custom build steps or tests ---
# add_test(...)
# add_custom_command(...)
