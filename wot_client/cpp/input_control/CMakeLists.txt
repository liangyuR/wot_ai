add_library(input_control STATIC
  input_control.cpp
)

target_include_directories(input_control PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Windows-specific libraries
if(WIN32)
  target_link_libraries(input_control PRIVATE
    user32
  )
endif()

