/**
 * 输入控制模块测试
 */

#include "input_control.h"

#include <chrono>
#include <iostream>
#include <thread>

int main() {
  std::cout << "==================================" << std::endl;
  std::cout << "Input Control Test" << std::endl;
  std::cout << "==================================" << std::endl;
  std::cout << "\nWARNING: This test will simulate keyboard/mouse input!"
            << std::endl;
  std::cout << "Make sure no important windows are in focus." << std::endl;
  std::cout << "\nPress Enter to continue (or Ctrl+C to cancel)..." << std::endl;
  std::cin.get();

  try {
    // 创建控制对象
    wot::InputControl control;
    std::cout << "✓ InputControl created" << std::endl;

    // 测试键盘输入
    std::cout << "\nTesting keyboard input..." << std::endl;
    std::cout << "  (Open Notepad to see the output)" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    control.TapKey('H');
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    control.TapKey('E');
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    control.TapKey('L');
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    control.TapKey('L');
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    control.TapKey('O');

    std::cout << "✓ Keyboard test completed (typed 'HELLO')" << std::endl;

    // 测试鼠标移动
    std::cout << "\nTesting mouse movement..." << std::endl;
    std::cout << "  (Watch your cursor move)" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    for (int i = 0; i < 5; ++i) {
      control.MoveMouse(50, 0);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "✓ Mouse movement test completed" << std::endl;

    // 测试性能
    std::cout << "\nTesting input performance..." << std::endl;
    const int num_inputs = 1000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_inputs; ++i) {
      control.PressKey('A');
      control.ReleaseKey('A');
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time = duration.count() / (double)num_inputs;
    std::cout << "✓ Performance test completed" << std::endl;
    std::cout << "  " << num_inputs << " inputs in " << duration.count() / 1000
              << " ms" << std::endl;
    std::cout << "  Average time per input: " << avg_time << " μs" << std::endl;

    if (avg_time < 1000) {
      std::cout << "\n✓ PASS: Input latency is excellent (< 1ms)" << std::endl;
    } else {
      std::cout << "\n⚠ WARNING: Input latency may be high" << std::endl;
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

