/**
 * Python bindings for C++ modules
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "input_control.h"
#include "screen_capture.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_bindings, m) {
  m.doc() = "World of Tanks AI C++ bindings";

  // ScreenCapture class
  py::class_<wot::ScreenCapture>(m, "ScreenCapture")
      .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
      .def("capture", &wot::ScreenCapture::Capture, "Capture entire screen")
      .def("capture_region", &wot::ScreenCapture::CaptureRegion,
           py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"),
           "Capture specific region")
      .def("get_fps", &wot::ScreenCapture::GetFps, "Get current FPS")
      .def("get_width", &wot::ScreenCapture::GetWidth, "Get screen width")
      .def("get_height", &wot::ScreenCapture::GetHeight, "Get screen height");

  // InputControl class
  py::class_<wot::InputControl>(m, "InputControl")
      .def(py::init<>())
      .def("press_key", &wot::InputControl::PressKey, py::arg("key"),
           "Press a key")
      .def("release_key", &wot::InputControl::ReleaseKey, py::arg("key"),
           "Release a key")
      .def("tap_key", &wot::InputControl::TapKey, py::arg("key"),
           py::arg("duration_ms") = 50, "Tap a key")
      .def("move_mouse", &wot::InputControl::MoveMouse, py::arg("dx"),
           py::arg("dy"), "Move mouse relatively")
      .def("move_mouse_absolute", &wot::InputControl::MoveMouseAbsolute,
           py::arg("x"), py::arg("y"), "Move mouse to absolute position")
      .def("mouse_click", &wot::InputControl::MouseClick,
           py::arg("button") = wot::MouseButton::kLeft, "Click mouse button")
      .def("release_all_keys", &wot::InputControl::ReleaseAllKeys,
           "Release all pressed keys");

  // MouseButton enum
  py::enum_<wot::MouseButton>(m, "MouseButton")
      .value("LEFT", wot::MouseButton::kLeft)
      .value("RIGHT", wot::MouseButton::kRight)
      .value("MIDDLE", wot::MouseButton::kMiddle)
      .export_values();
}

