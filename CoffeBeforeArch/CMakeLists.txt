cmake_minimum_required(VERSION 3.5)
project(Cuda VERSION 0.1.0 LANGUAGES C CXX)

# Set the path to the CUDA toolkit
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")

# Find CUDA package
find_package(CUDA REQUIRED)

# Add CUDA include and library directories
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(Cuda vectorAddition1.cu
        project.cpp
        project.h
        project.ui)
target_link_libraries(Cuda ${CUDA_LIBRARIES})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
