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

  // MouseButton enum - 必须先定义，因为 InputControl 要用到
  py::enum_<wot::MouseButton>(m, "MouseButton")
      .value("LEFT", wot::MouseButton::kLeft)
      .value("RIGHT", wot::MouseButton::kRight)
      .value("MIDDLE", wot::MouseButton::kMiddle)
      .export_values();

  // ScreenCapture class
  py::class_<wot::ScreenCapture>(m, "ScreenCapture")
      .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
      .def("Capture", &wot::ScreenCapture::Capture, "Capture entire screen")
      .def("CaptureRegion", &wot::ScreenCapture::CaptureRegion,
           py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"),
           "Capture specific region")
      .def("GetFps", &wot::ScreenCapture::GetFps, "Get current FPS")
      .def("GetWidth", &wot::ScreenCapture::GetWidth, "Get screen width")
      .def("GetHeight", &wot::ScreenCapture::GetHeight, "Get screen height");

  // InputControl class
  py::class_<wot::InputControl>(m, "InputControl")
      .def(py::init<>())
      .def("PressKey", &wot::InputControl::PressKey, py::arg("key"),
           "Press a key")
      .def("ReleaseKey", &wot::InputControl::ReleaseKey, py::arg("key"),
           "Release a key")
      .def("TapKey", &wot::InputControl::TapKey, py::arg("key"),
           py::arg("duration_ms") = 50, "Tap a key")
      .def("MoveMouse", &wot::InputControl::MoveMouse, py::arg("dx"),
           py::arg("dy"), "Move mouse relatively")
      .def("MoveMouseAbsolute", &wot::InputControl::MoveMouseAbsolute,
           py::arg("x"), py::arg("y"), "Move mouse to absolute position")
      .def("MouseClick", &wot::InputControl::MouseClick,
           py::arg("button") = wot::MouseButton::kLeft, "Click mouse button")
      .def("ReleaseAllKeys", &wot::InputControl::ReleaseAllKeys,
           "Release all pressed keys");
}

