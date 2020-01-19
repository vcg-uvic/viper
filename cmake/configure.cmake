find_package(CUDA REQUIRED)
set(COMMON_LIBS ${COMMON_LIBS} pthread)

find_package(Thrust REQUIRED)
include_directories(${THRUST_INCLUDE_DIR})

#set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIR})

add_subdirectory(${PROJECT_SOURCE_DIR}/libs/glfw)
set(OPENGP_HEADERONLY true)
add_definitions(-DOPENGP_HEADERONLY)
# add_subdirectory(${PROJECT_SOURCE_DIR}/libs/opengp)

find_package(Boost COMPONENTS thread REQUIRED)

find_package(GMP REQUIRED)

exec_program(${PROJECT_SOURCE_DIR}/find_cc.sh OUTPUT_VARIABLE COMPUTE_CAPABILITY)
message(STATUS "Detected Compute Capability: ${COMPUTE_CAPABILITY}")

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}
)

set(
    CMAKE_CXX_FLAGS
    "-std=c++11 -O3"
)