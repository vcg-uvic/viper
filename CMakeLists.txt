cmake_minimum_required (VERSION 3.2)

project (VIPER)

file(GLOB VIPER_SOURCES "Viper/*.cpp")
file(GLOB VIPER_CUDA_SOURCES "Viper/*.cu")
file(GLOB SPHEREWEIGHTS_SOURCES "SphereWeights/*.cpp")
file(GLOB DEMO_SOURCES "Demo/*.cu")
file(GLOB SPHEREWEIGHTS_CUDA_SOURCES "SphereWeights/*.cu")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(cmake/configure.cmake)

include_directories(
    ${PROJECT_SOURCE_DIR}/Viper
    ${PROJECT_SOURCE_DIR}/SphereWeights
    ${PROJECT_SOURCE_DIR}/libs/glm/
    ${PROJECT_SOURCE_DIR}/libs/eigen/
    ${PROJECT_SOURCE_DIR}/libs/rapidjson/include/
    ${PROJECT_SOURCE_DIR}/libs/opengp/src/
    ${PROJECT_SOURCE_DIR}/libs/igl/include/
    ${PROJECT_SOURCE_DIR}/libs/TetWild/
    ${PROJECT_SOURCE_DIR}/libs/geogram/src/lib/
    )


### ViperLib
cuda_add_library(ViperLib_static STATIC ${VIPER_SOURCES} ${VIPER_CUDA_SOURCES})
target_link_libraries(ViperLib_static ${COMMON_LIBS} CGAL Boost::thread gmp mpfr)

cuda_add_library(ViperLib SHARED ${VIPER_SOURCES} ${VIPER_CUDA_SOURCES})
target_link_libraries(ViperLib ${COMMON_LIBS} CGAL Boost::thread gmp mpfr)

### SphereWeights
add_executable(sphereweights ${SPHEREWEIGHTS_SOURCES})
target_include_directories(
    sphereweights
    PRIVATE
    ${PROJECT_SOURCE_DIR}/SphereWeights
)
target_link_libraries(sphereweights ${COMMON_LIBS} CGAL Boost::thread tbb gmp mpfr GLEW GL glfw png ViperLib)

### Demo
cuda_add_executable(demo ${DEMO_SOURCES})
target_link_libraries(demo ${COMMON_LIBS} CGAL Boost::thread tbb gmp mpfr GLEW GL glfw png ViperLib)
# configure_file(${PROJECT_SOURCE_DIR}/data/octopus.obj ${PROJECT_BINARY_DIR}/octopus.obj)
configure_file(${PROJECT_SOURCE_DIR}/data/texture.bin ${PROJECT_BINARY_DIR}/texture.bin COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/data/mesh.bin ${PROJECT_BINARY_DIR}/mesh.bin COPYONLY)

set(Viper_SO_PATH ${PROJECT_BINARY_DIR}/libViperLib.so)

