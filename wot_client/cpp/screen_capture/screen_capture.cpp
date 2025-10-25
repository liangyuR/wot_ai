/**
 * Screen capture module for World of Tanks AI
 * High-performance screen capture using Windows GDI/DirectX
 */

#include "screen_capture.h"

#include <chrono>
#include <stdexcept>

#ifdef _WIN32
#include <d3d11.h>
#include <dxgi1_2.h>
#include <windows.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif

namespace wot {

ScreenCapture::ScreenCapture(int width, int height)
    : width_(width), height_(height), initialized_(false) {
  Initialize();
}

ScreenCapture::~ScreenCapture() { Cleanup(); }

void ScreenCapture::Initialize() {
#ifdef _WIN32
  // Initialize DirectX for screen capture
  // TODO: Implement DirectX initialization
  // This provides much better performance than GDI
  
  initialized_ = true;
#else
  throw std::runtime_error("Screen capture only supported on Windows");
#endif
}

void ScreenCapture::Cleanup() {
  // Release DirectX resources
  initialized_ = false;
}

std::vector<uint8_t> ScreenCapture::Capture() {
  if (!initialized_) {
    throw std::runtime_error("Screen capture not initialized");
  }

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<uint8_t> buffer(width_ * height_ * 3);

#ifdef _WIN32
  // Get desktop window
  HWND desktop = GetDesktopWindow();
  HDC desktop_dc = GetDC(desktop);
  HDC mem_dc = CreateCompatibleDC(desktop_dc);

  // Create bitmap
  HBITMAP bitmap = CreateCompatibleBitmap(desktop_dc, width_, height_);
  HBITMAP old_bitmap = (HBITMAP)SelectObject(mem_dc, bitmap);

  // Copy screen to bitmap
  BitBlt(mem_dc, 0, 0, width_, height_, desktop_dc, 0, 0, SRCCOPY);

  // Get bitmap data
  BITMAPINFO bmi = {};
  bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  bmi.bmiHeader.biWidth = width_;
  bmi.bmiHeader.biHeight = -height_;  // Top-down
  bmi.bmiHeader.biPlanes = 1;
  bmi.bmiHeader.biBitCount = 24;
  bmi.bmiHeader.biCompression = BI_RGB;

  GetDIBits(mem_dc, bitmap, 0, height_, buffer.data(), &bmi, DIB_RGB_COLORS);

  // Cleanup
  SelectObject(mem_dc, old_bitmap);
  DeleteObject(bitmap);
  DeleteDC(mem_dc);
  ReleaseDC(desktop, desktop_dc);
#endif

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  last_capture_time_ms_ = duration.count();

  return buffer;
}

std::vector<uint8_t> ScreenCapture::CaptureRegion(int x, int y, int w, int h) {
  // TODO: Implement region capture for better performance
  // Capture only the game window region
  return Capture();
}

double ScreenCapture::GetFps() const {
  if (last_capture_time_ms_ <= 0) {
    return 0.0;
  }
  return 1000.0 / static_cast<double>(last_capture_time_ms_);
}

int ScreenCapture::GetWidth() const { return width_; }

int ScreenCapture::GetHeight() const { return height_; }

}  // namespace wot

