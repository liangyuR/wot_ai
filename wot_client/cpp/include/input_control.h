/**
 * Input control header
 */

#ifndef WOT_INPUT_CONTROL_H_
#define WOT_INPUT_CONTROL_H_

namespace wot {

/**
 * Mouse button enumeration
 */
enum class MouseButton { kLeft, kRight, kMiddle };

/**
 * Input control class for keyboard and mouse simulation
 */
class InputControl {
 public:
  /**
   * Constructor
   */
  InputControl();

  /**
   * Destructor
   */
  ~InputControl();

  /**
   * Press a key
   *
   * @param key Character key to press
   */
  void PressKey(char key);

  /**
   * Release a key
   *
   * @param key Character key to release
   */
  void ReleaseKey(char key);

  /**
   * Tap a key (press and release)
   *
   * @param key Character key to tap
   * @param duration_ms Duration to hold key (milliseconds)
   */
  void TapKey(char key, int duration_ms = 50);

  /**
   * Move mouse relatively
   *
   * @param dx X offset
   * @param dy Y offset
   */
  void MoveMouse(int dx, int dy);

  /**
   * Move mouse to absolute position
   *
   * @param x X coordinate
   * @param y Y coordinate
   */
  void MoveMouseAbsolute(int x, int y);

  /**
   * Click mouse button
   *
   * @param button Mouse button
   */
  void MouseClick(MouseButton button = MouseButton::kLeft);

  /**
   * Release all pressed keys
   */
  void ReleaseAllKeys();

 private:
  void Initialize();
  void Cleanup();
};

}  // namespace wot

#endif  // WOT_INPUT_CONTROL_H_

