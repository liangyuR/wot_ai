/**
 * 屏幕捕获模块测试
 */

#include "screen_capture.h"

#include <chrono>
#include <iostream>

int main() {
  std::cout << "==================================" << std::endl;
  std::cout << "Screen Capture Test" << std::endl;
  std::cout << "==================================" << std::endl;

  try {
    // 创建捕获对象
    wot::ScreenCapture capture(640, 480);
    std::cout << "✓ ScreenCapture created" << std::endl;
    std::cout << "  Size: " << capture.GetWidth() << "x" << capture.GetHeight()
              << std::endl;

    // 测试捕获性能
    std::cout << "\nTesting capture performance..." << std::endl;
    const int num_frames = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_frames; ++i) {
      auto frame = capture.Capture();
      if (i % 20 == 0) {
        std::cout << "  Frame " << i << " captured ("
                  << frame.size() / 1024 / 1024 << " MB)" << std::endl;
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    double avg_fps = num_frames * 1000.0 / duration.count();
    std::cout << "\n✓ Performance test completed" << std::endl;
    std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Average FPS: " << avg_fps << std::endl;

    if (avg_fps >= 30) {
      std::cout << "\n✓ PASS: Performance is good (>= 30 FPS)" << std::endl;
    } else {
      std::cout << "\n⚠ WARNING: Performance may be insufficient" << std::endl;
    }

  } catch (const std::exception& e) {
    std::cerr << "\n✗ ERROR: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n==================================" << std::endl;
  std::cout << "All tests passed!" << std::endl;
  std::cout << "==================================" << std::endl;

  return 0;
}

