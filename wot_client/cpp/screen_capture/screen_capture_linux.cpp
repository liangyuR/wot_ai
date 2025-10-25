/**
 * Linux screen capture implementation using X11
 * Note: This is a placeholder for Linux support
 * For production, use the Python fallback (mss library)
 */

#include "screen_capture.h"

#ifdef __linux__
#include <stdexcept>
#include <vector>
#include <cstdint>

// X11 headers (optional - will use Python fallback if not available)
#if __has_include(<X11/Xlib.h>)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#define HAS_X11 1
#else
#define HAS_X11 0
#endif

namespace wot {

#if HAS_X11
// X11 implementation (if available)
struct X11Data {
  Display* display;
  Window root;
  XImage* image;
  
  X11Data() : display(nullptr), root(0), image(nullptr) {}
  
  ~X11Data() {
    if (image) XDestroyImage(image);
    if (display) XCloseDisplay(display);
  }
};
#endif

ScreenCapture::ScreenCapture(int width, int height)
    : width_(width), height_(height), initialized_(false), last_capture_time_ms_(0) {
  Initialize();
}

ScreenCapture::~ScreenCapture() {
  Cleanup();
}

void ScreenCapture::Initialize() {
#if HAS_X11
  // Try to initialize X11
  Display* display = XOpenDisplay(nullptr);
  if (display) {
    XCloseDisplay(display);
    initialized_ = true;
    return;
  }
#endif
  
  // Fallback: Python implementation recommended
  throw std::runtime_error(
    "Linux screen capture requires X11 libraries.\n"
    "Install with: sudo apt install libx11-dev libxext-dev\n"
    "Or use Python fallback (mss library) for better compatibility."
  );
}

void ScreenCapture::Cleanup() {
  initialized_ = false;
}

std::vector<uint8_t> ScreenCapture::Capture() {
  if (!initialized_) {
    throw std::runtime_error("Screen capture not initialized");
  }

#if HAS_X11
  X11Data x11;
  x11.display = XOpenDisplay(nullptr);
  if (!x11.display) {
    throw std::runtime_error("Cannot open X display");
  }
  
  x11.root = DefaultRootWindow(x11.display);
  
  // Capture screen
  x11.image = XGetImage(
    x11.display,
    x11.root,
    0, 0,
    width_, height_,
    AllPlanes,
    ZPixmap
  );
  
  if (!x11.image) {
    throw std::runtime_error("Failed to capture screen");
  }
  
  std::vector<uint8_t> buffer(width_ * height_ * 3);
  
  // Convert to RGB
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      unsigned long pixel = XGetPixel(x11.image, x, y);
      int idx = (y * width_ + x) * 3;
      buffer[idx + 0] = (pixel & 0xFF0000) >> 16;  // R
      buffer[idx + 1] = (pixel & 0x00FF00) >> 8;   // G
      buffer[idx + 2] = (pixel & 0x0000FF);        // B
    }
  }
  
  return buffer;
#else
  throw std::runtime_error("X11 not available - use Python fallback");
#endif
}

std::vector<uint8_t> ScreenCapture::CaptureRegion(int x, int y, int w, int h) {
  // Simplified implementation
  return Capture();
}

double ScreenCapture::GetFps() const {
  if (last_capture_time_ms_ <= 0) {
    return 0.0;
  }
  return 1000.0 / static_cast<double>(last_capture_time_ms_);
}

int ScreenCapture::GetWidth() const {
  return width_;
}

int ScreenCapture::GetHeight() const {
  return height_;
}

}  // namespace wot

#endif  // __linux__

