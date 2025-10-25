add_library(screen_capture STATIC
  screen_capture.cpp
)

target_include_directories(screen_capture PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Windows-specific libraries
if(WIN32)
  target_link_libraries(screen_capture PRIVATE
    gdi32
    user32
    d3d11
    dxgi
  )
endif()

