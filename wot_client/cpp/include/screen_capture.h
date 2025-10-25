/**
 * Screen capture header
 */

#ifndef WOT_SCREEN_CAPTURE_H_
#define WOT_SCREEN_CAPTURE_H_

#include <cstdint>
#include <vector>

namespace wot {

/**
 * High-performance screen capture class
 */
class ScreenCapture {
 public:
  /**
   * Constructor
   *
   * @param width Screen width
   * @param height Screen height
   */
  ScreenCapture(int width, int height);

  /**
   * Destructor
   */
  ~ScreenCapture();

  /**
   * Capture entire screen
   *
   * @return RGB buffer (width * height * 3 bytes)
   */
  std::vector<uint8_t> Capture();

  /**
   * Capture specific region
   *
   * @param x X coordinate
   * @param y Y coordinate
   * @param w Width
   * @param h Height
   * @return RGB buffer
   */
  std::vector<uint8_t> CaptureRegion(int x, int y, int w, int h);

  /**
   * Get current FPS
   */
  double GetFps() const;

  /**
   * Get screen width
   */
  int GetWidth() const;

  /**
   * Get screen height
   */
  int GetHeight() const;

 private:
  void Initialize();
  void Cleanup();

  int width_;
  int height_;
  bool initialized_;
  int64_t last_capture_time_ms_;
};

}  // namespace wot

#endif  // WOT_SCREEN_CAPTURE_H_

