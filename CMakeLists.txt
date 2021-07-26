cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(predator-prey)

# Include what you use detector
find_program(IWYU NAMES include-what-you-use)
if(IWYU)
  message(STATUS "executing include-what-you-use")
  set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU})
endif()

# Set parameters
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# Need to link with intel on mac since there is limited support for arm64 macos
# set(CMAKE_OSX_ARCHITECTURES x86_64)

# Add the executable
file(GLOB SOURCES src/*.cpp include/*.h)
add_executable(${PROJECT_NAME} ${SOURCES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_include_directories(${PROJECT_NAME} PUBLIC src include)

target_compile_options(
  ${PROJECT_NAME}
  PUBLIC -Werror
         -Wall
         -Wextra
         -Oz
         -flto
         -std=c++17
         -pthread
         -ffunction-sections
         -fdata-sections)

add_custom_command(
  TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND du -h "${CMAKE_CURRENT_SOURCE_DIR}/build/${PROJECT_NAME}")

# Link

# torch
find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

# robosim
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/robosim)
find_package(ROBOSIM REQUIRED)
target_link_libraries(${PROJECT_NAME} ROBOSIM::Core)

# Link

# The following code block is suggested to be used on Windows. According to
# https://github.com/pytorch/pytorch/issues/25457, the DLLs need to be copied to
# avoid memory errors.
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll
    ")
  add_custom_command(
    TARGET example-app
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example-app>)
endif(MSVC)
