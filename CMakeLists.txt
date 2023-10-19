cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 52;61;75;86;89)

project(marching_cubes_cuda LANGUAGES CXX CUDA)

list(APPEND EXTRA_INCLUDES include ${CMAKE_BINARY_DIR})

set(SOURCES src/main.cu src/marching_cubes.cu)

add_executable(main ${SOURCES})

target_include_directories(main PUBLIC ${EXTRA_INCLUDES})
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
if(CMAKE_BUILD_TYPE STREQUAL Debug)
    target_compile_options(main PUBLIC -G)
elseif(CMAKE_BUILD_TYPE STREQUAL Release)
    target_compile_options(main PUBLIC -w)
endif()

install(FILES ${SOURCES} DESTINATION src)
set(HEADS include/marching_cubes.cuh include/types.cuh include/extra_arrays.cuh include/helper_cuda.h include/helper_string.h)
install(FILES ${HEADS} DESTINATION include)
install(FILES CMakeLists.txt DESTINATION .)

include(InstallRequiredSystemLibraries)
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)
