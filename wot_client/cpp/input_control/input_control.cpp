/**
 * Input control module for World of Tanks AI
 * Simulates keyboard and mouse input
 */

#include "input_control.h"

#include <stdexcept>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace wot {

InputControl::InputControl() { Initialize(); }

InputControl::~InputControl() { Cleanup(); }

void InputControl::Initialize() {
  // Initialize input system
}

void InputControl::Cleanup() {
  // Release all pressed keys
  ReleaseAllKeys();
}

void InputControl::PressKey(char key) {
#ifdef _WIN32
  INPUT input = {};
  input.type = INPUT_KEYBOARD;
  input.ki.wVk = VkKeyScanA(key);
  input.ki.dwFlags = 0;  // Key down

  SendInput(1, &input, sizeof(INPUT));
#endif
}

void InputControl::ReleaseKey(char key) {
#ifdef _WIN32
  INPUT input = {};
  input.type = INPUT_KEYBOARD;
  input.ki.wVk = VkKeyScanA(key);
  input.ki.dwFlags = KEYEVENTF_KEYUP;

  SendInput(1, &input, sizeof(INPUT));
#endif
}

void InputControl::TapKey(char key, int duration_ms) {
  PressKey(key);
  std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
  ReleaseKey(key);
}

void InputControl::MoveMouse(int dx, int dy) {
#ifdef _WIN32
  INPUT input = {};
  input.type = INPUT_MOUSE;
  input.mi.dx = dx;
  input.mi.dy = dy;
  input.mi.dwFlags = MOUSEEVENTF_MOVE;

  SendInput(1, &input, sizeof(INPUT));
#endif
}

void InputControl::MoveMouseAbsolute(int x, int y) {
#ifdef _WIN32
  // Convert to normalized coordinates (0-65535)
  int screen_width = GetSystemMetrics(SM_CXSCREEN);
  int screen_height = GetSystemMetrics(SM_CYSCREEN);

  int normalized_x = (x * 65535) / screen_width;
  int normalized_y = (y * 65535) / screen_height;

  INPUT input = {};
  input.type = INPUT_MOUSE;
  input.mi.dx = normalized_x;
  input.mi.dy = normalized_y;
  input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;

  SendInput(1, &input, sizeof(INPUT));
#endif
}

void InputControl::MouseClick(MouseButton button) {
#ifdef _WIN32
  INPUT input = {};
  input.type = INPUT_MOUSE;

  DWORD down_flag = 0;
  DWORD up_flag = 0;

  switch (button) {
    case MouseButton::kLeft:
      down_flag = MOUSEEVENTF_LEFTDOWN;
      up_flag = MOUSEEVENTF_LEFTUP;
      break;
    case MouseButton::kRight:
      down_flag = MOUSEEVENTF_RIGHTDOWN;
      up_flag = MOUSEEVENTF_RIGHTUP;
      break;
    case MouseButton::kMiddle:
      down_flag = MOUSEEVENTF_MIDDLEDOWN;
      up_flag = MOUSEEVENTF_MIDDLEUP;
      break;
  }

  // Press
  input.mi.dwFlags = down_flag;
  SendInput(1, &input, sizeof(INPUT));

  // Small delay
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Release
  input.mi.dwFlags = up_flag;
  SendInput(1, &input, sizeof(INPUT));
#endif
}

void InputControl::ReleaseAllKeys() {
  // Release common game keys
  const char keys[] = {'W', 'A', 'S', 'D', 'Q', 'E', 'R', ' '};
  for (char key : keys) {
    ReleaseKey(key);
  }
}

}  // namespace wot

