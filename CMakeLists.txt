cmake_minimum_required(VERSION 3.30)

project(
    ml_training_perf
    VERSION 0.0.1
    LANGUAGES CXX
)

option(
    MLTP_ENABLE_CUDA
    "Build CUDA implementations"
    ON
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

find_package(
    Python3
    REQUIRED
    COMPONENTS
        Interpreter
        Development.Module
)

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

if(MLTP_ENABLE_CUDA)
    enable_language(CUDA)

    find_package(
        CUDAToolkit
        REQUIRED
    )

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(
            CMAKE_CUDA_ARCHITECTURES
            89
            CACHE STRING
            "CUDA architectures to compile"
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# PyTorch
#
# Ask the selected Python interpreter where its PyTorch CMake package lives.
# This avoids hardcoding /home/joshp/miniforge3/... into the repository.
# ---------------------------------------------------------------------------

execute_process(
    COMMAND
        "${Python3_EXECUTABLE}"
        -c
        "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE
        MLTP_TORCH_PREFIX_RESULT
    OUTPUT_VARIABLE
        MLTP_TORCH_PREFIX_PATH
    ERROR_VARIABLE
        MLTP_TORCH_PREFIX_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT MLTP_TORCH_PREFIX_RESULT EQUAL 0)
    message(
        FATAL_ERROR
        "Could not determine the PyTorch CMake prefix using "
        "${Python3_EXECUTABLE}:\n"
        "${MLTP_TORCH_PREFIX_ERROR}"
    )
endif()

list(
    PREPEND
    CMAKE_PREFIX_PATH
    "${MLTP_TORCH_PREFIX_PATH}"
)

find_package(
    Torch
    REQUIRED
)

message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "PyTorch CMake prefix: ${MLTP_TORCH_PREFIX_PATH}")
message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "CUDA enabled: ${MLTP_ENABLE_CUDA}")

# PyTorch publishes required ABI/compiler options through TORCH_CXX_FLAGS.
separate_arguments(
    MLTP_TORCH_COMPILE_OPTIONS
    NATIVE_COMMAND
    "${TORCH_CXX_FLAGS}"
)

# ---------------------------------------------------------------------------
# Extension target
# ---------------------------------------------------------------------------

Python3_add_library(
    _C
    MODULE
    WITH_SOABI
    cpp/src/bindings.cpp
    cpp/src/scaled_relu_cpu.cpp
)

if(MLTP_ENABLE_CUDA)
    target_sources(
        _C
        PRIVATE
            cuda/scaled_relu_cuda.cu
    )
endif()

target_include_directories(
    _C
    PRIVATE
        "${PROJECT_SOURCE_DIR}/cpp/include"
)

target_link_libraries(
    _C
    PRIVATE
        ${TORCH_LIBRARIES}
        Python3::Module
)

if(MLTP_ENABLE_CUDA)
    target_link_libraries(
        _C
        PRIVATE
            CUDA::cudart
    )
endif()

target_compile_definitions(
    _C
    PRIVATE
        TORCH_EXTENSION_NAME=_C
)

target_compile_options(
    _C
    PRIVATE
        ${MLTP_TORCH_COMPILE_OPTIONS}

        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wextra>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wpedantic>

        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>>:-fno-omit-frame-pointer>

        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(
    _C
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        PREFIX ""
        OUTPUT_NAME "_C"
)

if(MLTP_ENABLE_CUDA)
    set_target_properties(
        _C
        PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED YES
            CUDA_EXTENSIONS NO
    )
endif()

# scikit-build-core copies CMake's install tree into the wheel/editable
# installation. The extension therefore belongs under the Python package.
install(
    TARGETS
        _C
    LIBRARY DESTINATION
        ml_training_perf
    RUNTIME DESTINATION
        ml_training_perf
)