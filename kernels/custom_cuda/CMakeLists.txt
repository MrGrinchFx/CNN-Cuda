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

# --- Set C++ Standard ---
set(CMAKE_CXX_STANDARD 17) # Or 20, 23 etc., depending on your project's needs
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard-compliant C++

# --- Source Files ---
# Add your C++ source files
#set(CXX_SOURCES
#    main.cpp
#    neuralnet.cpp
#)

# Find all .cu and .cuh files in Data and Layers directories
# glob_recursive is useful here if you have subdirectories within Data/Layers
file(GLOB_RECURSE CUDA_SOURCES
    Data/*.cu
    Layers/*.cu
    main.cu
    neuralnet.cu
)

# Combine all source files
set(ALL_SOURCES
    ${CUDA_SOURCES}
)

# --- Create Executable ---
# IMPORTANT: Define the target *before* setting its properties or linking
add_executable(${PROJECT_NAME} ${ALL_SOURCES})

# --- CUDA Specific Settings (on the target) ---
# Specify the GPU architecture(s) your kernels will target.
# IMPORTANT: Adjust this to your target GPU's compute capability.
set(CUDA_ARCH_CODE "86") # <--- CHANGE THIS TO YOUR GPU'S ARCHITECTURE!
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH_CODE})


# --- Include Directories ---
# Add include directories for your header files (.hpp, .cuh)
# CMake will automatically add the CUDA include path because LANGUAGES CXX CUDA is set.
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} # For main.cpp, neuralnet.cpp, utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Data # For headers in Data/
    ${CMAKE_CURRENT_SOURCE_DIR}/Layers # For headers in Layers/
)

# --- Link Libraries ---
# Link against the CUDA runtime library. This is crucial.
# Using CUDA_LIBRARIES variable as a fallback, which is set by the CUDA language module.
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})

