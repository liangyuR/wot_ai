cmake_minimum_required(VERSION 3.15)
project(WotAI CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags (Google C++ Style)
if(MSVC)
  add_compile_options(/W4 /WX)
  add_compile_options(/O2)  # Optimization
else()
  add_compile_options(-Wall -Wextra -Werror)
  add_compile_options(-O3)  # Optimization
endif()

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found via CMake, trying pip install location")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  find_package(pybind11 CONFIG REQUIRED)
endif()

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Python3_INCLUDE_DIRS}
)

# Screen capture module
add_subdirectory(screen_capture)

# Input control module
add_subdirectory(input_control)

# Python bindings
pybind11_add_module(cpp_bindings
  bindings/bindings.cpp
)

target_link_libraries(cpp_bindings PRIVATE
  screen_capture
  input_control
)

# Install
install(TARGETS cpp_bindings
  LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../python
)

