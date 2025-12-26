#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NavigationRuntime - Long-lived architecture with session-based lifecycle.

Key design:
- Models (MinimapDetector) and UI (DpgNavDebugView) are created once and reused across matches
- Per-match state is managed by NavigationSession and reset between matches
- Threads can be paused/resumed without destroying resources
- CUDA cache is cleaned at match end to prevent memory leaks
"""

import gc
import math
import random
import threading
import time
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger

from src.gui.debug_view import DpgNavDebugView
from src.navigation.controller.movement_service import MovementService
from src.navigation.nav_runtime.navigation_session import NavigationSession
from src.navigation.nav_runtime.path_follower_wrapper import PathFollowerWrapper
from src.navigation.nav_runtime.path_planner_wrapper import PathPlannerWrapper
from src.navigation.nav_runtime.stuck_detector import StuckDetector
from src.navigation.path_planner.path_planning_service import PathPlanningService
from src.navigation.service.capture_service import CaptureService
from src.navigation.service.data_hub import DataHub
from src.utils.global_path import GetGlobalConfig
from src.vision.map_name_detector import MapNameDetector
from src.vision.minimap_anchor_detector import MinimapAnchorDetector
from src.vision.minimap_detector import MinimapDetector


class FrameBuffer:
    """Thread-safe latest-frame buffer (single producer, single consumer)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._frame_event = threading.Event()

    def put(self, frame: np.ndarray) -> None:
        """Write the latest frame (capture thread)."""
        with self._lock:
            self._frame = frame
            self._timestamp = time.perf_counter()
            self._frame_event.set()

    def get(self, timeout: float = 0.05) -> Tuple[Optional[np.ndarray], float]:
        """Get the latest frame and timestamp (detection thread)."""
        if not self._frame_event.wait(timeout):
            return None, self._timestamp
        with self._lock:
            frame = self._frame
            ts = self._timestamp
            self._frame_event.clear()
            return frame, ts

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._frame = None
            self._timestamp = 0.0
            self._frame_event.clear()


class NavigationRuntime:
    """Navigation runtime with long-lived model/UI and session-based control.
    
    Lifecycle:
        UNINITIALIZED -> init() -> IDLE
        IDLE -> startSession() -> RUNNING
        RUNNING -> pauseSession() -> PAUSED
        PAUSED -> resumeSession() -> RUNNING
        RUNNING/PAUSED -> endSession() -> IDLE
        IDLE -> shutdown() -> (destroyed)
    """

    class State(Enum):
        UNINITIALIZED = 0  # Not initialized yet
        IDLE = 1           # Models loaded, threads not running
        RUNNING = 2        # Session active, threads running
        PAUSED = 3         # Session paused, threads sleeping

    def __init__(self):
        self.cfg_ = GetGlobalConfig()
        self._state = NavigationRuntime.State.UNINITIALIZED
        
        # Long-lived resources (created in init, destroyed in shutdown)
        self.capture_: Optional[CaptureService] = None
        self.view_: Optional[DpgNavDebugView] = None
        self.planner_service_: Optional[PathPlanningService] = None
        self.data_hub_: Optional[DataHub] = None
        self.minimap_detector_: Optional[MinimapDetector] = None
        self.minimap_anchor_detector_: Optional[MinimapAnchorDetector] = None
        self.minimap_name_detector_: Optional[MapNameDetector] = None
        self.move_: Optional[MovementService] = None
        self.path_follower_wrapper_: Optional[PathFollowerWrapper] = None
        self.stuck_detector_: Optional[StuckDetector] = None
        
        # Per-session state
        self.session_: Optional[NavigationSession] = None
        self.path_planner_: Optional[PathPlannerWrapper] = None
        
        # Cached minimap region (detected once in init, reused across sessions)
        self.minimap_region_: Optional[dict] = None
        
        # Thread management
        self._threads_running = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        
        self._capture_thread: Optional[threading.Thread] = None
        self._det_thread: Optional[threading.Thread] = None
        self._ctrl_thread: Optional[threading.Thread] = None
        self._view_thread: Optional[threading.Thread] = None
        self._frame_buffer: Optional[FrameBuffer] = None
        
        # Config cache
        self.reverse_duration_s_ = self.cfg_.stuck_detection.reverse_duration_s
        self.max_stuck_count_ = self.cfg_.stuck_detection.max_stuck_count

    # =========================================================================
    # Lifecycle: init / shutdown (one-time)
    # =========================================================================
    
    def init(self) -> bool:
        """Initialize long-lived resources (models, UI). Call once at program start.
        
        Returns:
            True if initialization succeeded, False otherwise.
        """
        if self._state != NavigationRuntime.State.UNINITIALIZED:
            logger.warning("NavigationRuntime already initialized")
            return True
        
        logger.info("Initializing NavigationRuntime...")
        
        try:
            # Create long-lived components
            self.capture_ = CaptureService(self.cfg_.monitor_index)
            self.planner_service_ = PathPlanningService(self.cfg_)
            self.data_hub_ = DataHub()
            
            self.minimap_detector_ = MinimapDetector()
            self.minimap_anchor_detector_ = MinimapAnchorDetector()
            self.minimap_name_detector_ = MapNameDetector()
            
            self.move_ = MovementService()
            self.path_follower_wrapper_ = PathFollowerWrapper()
            self.stuck_detector_ = StuckDetector()
            
            # Load models (this takes time, do it once)
            if not self.minimap_detector_.LoadModel():
                logger.error("Failed to load minimap detection model")
                self._cleanupLongLivedResources()
                return False
            
            # Create UI if enabled (but don't start the loop yet)
            if self.cfg_.ui.enable:
                self.view_ = DpgNavDebugView()
            
            self._state = NavigationRuntime.State.IDLE
            logger.info("NavigationRuntime initialized (IDLE)")
            return True
            
        except Exception as e:
            logger.error(f"NavigationRuntime init failed: {e}")
            self._cleanupLongLivedResources()
            return False
    
    def detectMinimapRegion(self) -> bool:
        """Detect and cache minimap region. Call once when game screen is ready.
        
        This should be called after init() when the game screen is visible.
        The detected region is cached and reused across all sessions.
        
        Returns:
            True if detection succeeded.
        """
        if self._state == NavigationRuntime.State.UNINITIALIZED:
            logger.error("Call init() before detectMinimapRegion()")
            return False
        
        logger.info("Detecting minimap region...")
        first_frame = self.capture_.grab()
        if first_frame is None:
            logger.error("Screen grab failed")
            return False
        
        self.minimap_region_ = self.minimap_anchor_detector_.DetectRegion(first_frame)
        if not self.minimap_region_:
            logger.error("Failed to detect minimap region")
            return False
        
        logger.info(f"Minimap region cached: {self.minimap_region_}")
        return True
    
    def shutdown(self) -> None:
        """Destroy all resources. Call once at program exit."""
        if self._state == NavigationRuntime.State.UNINITIALIZED:
            return
        
        logger.info("Shutting down NavigationRuntime...")
        
        # End any active session first
        if self._state in (NavigationRuntime.State.RUNNING, NavigationRuntime.State.PAUSED):
            self.endSession()
        
        # Cleanup long-lived resources
        self._cleanupLongLivedResources()
        
        # Final CUDA cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self._state = NavigationRuntime.State.UNINITIALIZED
        logger.info("NavigationRuntime shutdown complete")
    
    def _cleanupLongLivedResources(self) -> None:
        """Cleanup all long-lived resources."""
        # Close UI
        if self.view_ is not None:
            try:
                self.view_.close()
            except Exception as e:
                logger.debug(f"Error closing view: {e}")
            self.view_ = None
        
        # Cleanup detector models
        if self.minimap_detector_ is not None:
            try:
                self.minimap_detector_.Cleanup()
            except Exception as e:
                logger.debug(f"Error cleaning up minimap_detector: {e}")
            self.minimap_detector_ = None
        
        # Clear other references
        self.capture_ = None
        self.planner_service_ = None
        self.data_hub_ = None
        self.minimap_anchor_detector_ = None
        self.minimap_name_detector_ = None
        self.move_ = None
        self.path_follower_wrapper_ = None
        self.stuck_detector_ = None

    # =========================================================================
    # Session lifecycle: startSession / endSession / pauseSession / resumeSession
    # =========================================================================
    
    def startSession(self, map_name: Optional[str] = None) -> bool:
        """Start a new game session. Call at match start.
        
        Args:
            map_name: Optional map name. If None, will auto-detect.
            
        Returns:
            True if session started successfully.
        """
        if self._state == NavigationRuntime.State.UNINITIALIZED:
            logger.error("NavigationRuntime not initialized. Call init() first.")
            return False
        
        if self._state == NavigationRuntime.State.RUNNING:
            logger.warning("Session already running")
            return True
        
        if self._state == NavigationRuntime.State.PAUSED:
            logger.info("Resuming paused session...")
            self.resumeSession()
            return True
        
        # State is IDLE, start new session
        logger.info("Starting new navigation session...")
        
        # Create session state
        self.session_ = NavigationSession()
        
        # Use cached minimap region, or detect if not cached
        if self.minimap_region_ is None:
            logger.info("Minimap region not cached, detecting...")
            if not self.detectMinimapRegion():
                logger.error("Failed to detect minimap region")
                self.session_ = None
                return False
        
        self.session_.minimap_region_ = self.minimap_region_
        logger.info(f"Using minimap region: {self.minimap_region_}")
        
        # Auto-detect map name if not provided
        if not map_name:
            from src.utils.key_controller import KeyController
            key_controller = KeyController()
            key_controller.press("b")
            time.sleep(2)
            map_name = self.minimap_name_detector_.detect()
            key_controller.release("b")
            
            if not map_name:
                logger.error("Failed to detect map name")
                self.session_ = None
                return False
        
        self.session_.map_name_ = map_name
        logger.info(f"Map name: {map_name}")
        
        # Load map for planner
        mmap_size = self.session_.getMinimapSize()
        if not self.planner_service_.load_map(map_name, mmap_size):
            logger.error("Failed to load map for planner")
            self.session_ = None
            return False
        
        # Calculate scale factors
        grid_w, grid_h = self.cfg_.grid.size
        self.session_.scale_x_ = mmap_size[0] / grid_w
        self.session_.scale_y_ = mmap_size[1] / grid_h
        
        # Create path planner wrapper
        self.path_planner_ = PathPlannerWrapper(
            planner=self.planner_service_,
            minimap_size=mmap_size,
            scale_xy=(self.session_.scale_x_, self.session_.scale_y_),
        )
        
        # Update UI grid mask
        if self.view_ is not None:
            self.view_.set_grid_mask(self.planner_service_.get_grid_mask())
        
        # Reset per-session components
        self.stuck_detector_.reset()
        self.stuck_detector_.resetStuckCount()
        self.data_hub_.reset()
        self.minimap_detector_.Reset()
        
        # Start threads
        self._startThreads()
        
        self._state = NavigationRuntime.State.RUNNING
        logger.info(f"Navigation session started: {map_name}")
        return True
    
    def endSession(self) -> None:
        """End the current game session. Call at match end."""
        if self._state not in (NavigationRuntime.State.RUNNING, NavigationRuntime.State.PAUSED):
            logger.debug("No active session to end")
            return
        
        logger.info("Ending navigation session...")
        
        # Resume if paused (to allow threads to exit)
        if self._state == NavigationRuntime.State.PAUSED:
            self._pause_event.set()
        
        # Stop movement
        try:
            self.move_.stop()
        except Exception:
            pass
        
        # Stop threads
        self._stopThreads()
        
        # Reset session-level state in detector (clears tracker history)
        if self.minimap_detector_ is not None:
            self.minimap_detector_.resetSession()
        
        # Reset other session state
        if self.stuck_detector_ is not None:
            self.stuck_detector_.reset()
            self.stuck_detector_.resetStuckCount()
        
        if self.data_hub_ is not None:
            self.data_hub_.reset()
        
        # Clear session
        self.session_ = None
        self.path_planner_ = None
        
        self._state = NavigationRuntime.State.IDLE
        logger.info("Navigation session ended (IDLE)")
    
    def pauseSession(self) -> None:
        """Pause the current session. Threads will sleep but stay alive."""
        if self._state != NavigationRuntime.State.RUNNING:
            logger.debug("Cannot pause: not in RUNNING state")
            return
        
        logger.info("Pausing navigation session...")
        
        # Stop movement
        try:
            self.move_.stop()
        except Exception:
            pass
        
        # Pause threads
        self._pause_event.clear()
        
        self._state = NavigationRuntime.State.PAUSED
        logger.info("Navigation session paused")
    
    def resumeSession(self) -> None:
        """Resume a paused session."""
        if self._state != NavigationRuntime.State.PAUSED:
            logger.debug("Cannot resume: not in PAUSED state")
            return
        
        logger.info("Resuming navigation session...")
        
        # Resume threads
        self._pause_event.set()
        
        self._state = NavigationRuntime.State.RUNNING
        logger.info("Navigation session resumed")

    # =========================================================================
    # State queries
    # =========================================================================
    
    def getState(self) -> "NavigationRuntime.State":
        """Get current runtime state."""
        return self._state
    
    def isRunning(self) -> bool:
        """Check if a session is actively running (not paused)."""
        return self._state == NavigationRuntime.State.RUNNING
    
    def isInitialized(self) -> bool:
        """Check if runtime has been initialized."""
        return self._state != NavigationRuntime.State.UNINITIALIZED

    # =========================================================================
    # Thread management
    # =========================================================================
    
    def _startThreads(self) -> None:
        """Start capture, detection, control, and UI threads."""
        if self._threads_running:
            return
        
        self._threads_running = True
        self._pause_event.set()
        self._frame_buffer = FrameBuffer()
        
        self._capture_thread = threading.Thread(target=self._captureLoop, daemon=True)
        self._det_thread = threading.Thread(target=self._detLoop, daemon=True)
        self._ctrl_thread = threading.Thread(target=self._ctrlLoop, daemon=True)
        
        self._capture_thread.start()
        self._det_thread.start()
        self._ctrl_thread.start()
        
        # Start UI thread if view exists and not already running
        if self.view_ is not None and (self._view_thread is None or not self._view_thread.is_alive()):
            self._view_thread = threading.Thread(target=self.view_.run, daemon=True)
            self._view_thread.start()
        
        logger.debug("Worker threads started")
    
    def _stopThreads(self) -> None:
        """Stop all worker threads."""
        if not self._threads_running:
            return
        
        self._threads_running = False
        
        # Wait for threads to exit
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self._det_thread and self._det_thread.is_alive():
            self._det_thread.join(timeout=1.0)
        if self._ctrl_thread and self._ctrl_thread.is_alive():
            self._ctrl_thread.join(timeout=1.0)
        
        # Clear frame buffer
        if self._frame_buffer is not None:
            self._frame_buffer.clear()
        
        self._capture_thread = None
        self._det_thread = None
        self._ctrl_thread = None
        
        logger.debug("Worker threads stopped")

    # =========================================================================
    # Thread loops
    # =========================================================================
    
    def _captureLoop(self) -> None:
        """Capture thread: grab minimap region frames."""
        logger.info("Capture thread started")
        
        while self._threads_running:
            # Check pause
            if not self._pause_event.wait(timeout=0.1):
                continue
            
            if self.session_ is None or not self.session_.hasValidRegion():
                time.sleep(0.01)
                continue
            
            try:
                x, y = self.session_.getMinimapOffset()
                w, h = self.session_.getMinimapSize()
                frame = self.capture_.grab_region(x, y, w, h)
                if frame is not None:
                    self._frame_buffer.put(frame)
            except Exception as e:
                logger.error(f"Capture thread error: {e}")
                time.sleep(0.01)
        
        logger.info("Capture thread exited")
    
    def _detLoop(self) -> None:
        """Detection thread: run YOLO detection on frames."""
        logger.info("Detection thread started")
        
        det_fps = self.cfg_.detection.max_fps
        min_interval = 1.0 / det_fps
        
        view_update_interval = 0
        view_update_counter = 0
        last_processed_ts = 0.0
        
        while self._threads_running:
            # Check pause
            if not self._pause_event.wait(timeout=0.1):
                continue
            
            loop_start = time.perf_counter()
            
            try:
                frame, ts = self._frame_buffer.get(timeout=0.05)
                if frame is None or ts <= last_processed_ts:
                    continue
                last_processed_ts = ts
                
                # Update view with frame
                view_update_counter += 1
                if self.view_ is not None and view_update_counter >= view_update_interval:
                    self.view_.update_minimap_frame(frame)
                    view_update_counter = 0
                
                # Run detection
                det = self.minimap_detector_.Detect(frame)
                if det is None or getattr(det, "self_pos", None) is None:
                    continue
                
                self.data_hub_.set_latest_detection(det)
                
            except Exception as e:
                logger.error(f"Detection thread error: {e}")
                time.sleep(0.01)
            finally:
                elapsed = time.perf_counter() - loop_start
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
        
        logger.info("Detection thread exited")
    
    def _ctrlLoop(self) -> None:
        """Control thread: path planning and movement control."""
        logger.info("Control thread started")
        
        ctrl_fps = 30
        interval = 1.0 / ctrl_fps
        view_update_interval = 3
        view_update_counter = 0
        
        while self._threads_running:
            # Check pause
            if not self._pause_event.wait(timeout=0.1):
                continue
            
            if self.session_ is None:
                time.sleep(interval)
                continue
            
            t0 = time.perf_counter()
            
            # Get latest detection
            det = self.data_hub_.get_latest_detection(max_age=1.0)
            has_detection = det is not None and getattr(det, "self_pos", None) is not None
            
            self.move_.update_detection_status(has_detection)
            
            if not has_detection:
                should_blind_forward = self.move_.tick_blind_forward()
                if should_blind_forward:
                    dt = time.perf_counter() - t0
                    if dt < interval:
                        time.sleep(interval - dt)
                    continue
                time.sleep(interval)
                continue
            
            pos = det.self_pos
            heading = math.radians(getattr(det, "self_angle", 0.0) or 0.0)
            
            # Check if stuck
            is_stuck = self.stuck_detector_.update(pos)
            need_replan = (
                not self.session_.current_path_world_
                or is_stuck
                or self.session_.current_target_idx_ >= len(self.session_.current_path_world_) - 1
            )
            
            # Replan if needed
            if need_replan:
                self.move_.stop()
                
                if is_stuck:
                    self.stuck_detector_.incrementStuckCount()
                    stuck_count = self.stuck_detector_.getStuckCount()
                    
                    if stuck_count >= self.max_stuck_count_:
                        turn_bias = random.uniform(-0.6, 0.6)
                        logger.warning(
                            f"Stuck {stuck_count} times, random turn (bias={turn_bias:.2f})"
                        )
                    else:
                        turn_bias = 0.0
                        logger.info(f"Stuck #{stuck_count}, reversing")
                    
                    self.move_.reverse(
                        duration_s=self.reverse_duration_s_,
                        turn_bias=turn_bias,
                    )
                
                grid_path, world_path = self.planner_service_.plan_path(det)
                
                if not world_path:
                    self.move_.stop()
                    time.sleep(interval)
                    continue
                
                self.session_.current_path_grid_ = grid_path
                self.session_.current_path_world_ = world_path
                self.session_.current_target_idx_ = 0
                self.session_.last_published_idx_ = -1
                
                try:
                    self.data_hub_.set_current_path(
                        grid_path=grid_path,
                        world_path=world_path,
                        target_idx=0,
                    )
                except Exception:
                    pass
                
                self.stuck_detector_.reset()
                if not is_stuck:
                    self.stuck_detector_.resetStuckCount()
                
                logger.info(f"New path: {len(world_path)} nodes")
            
            # Follow path
            follow_result = self.path_follower_wrapper_.follow(
                current_pos=pos,
                path_world=self.session_.current_path_world_,
                current_target_idx=self.session_.current_target_idx_,
            )
            
            target_world = follow_result.target_world
            dev = follow_result.distance_to_path
            dist_goal = follow_result.distance_to_goal
            goal_reached = follow_result.goal_reached
            self.session_.current_target_idx_ = follow_result.current_idx
            
            # Update view
            view_update_counter += 1
            if self.view_ is not None and view_update_counter >= view_update_interval:
                view_update_counter = 0
                try:
                    goal_pos = getattr(det, "goal_pos", None)
                    self.view_.update_nav_state(
                        self_pos_mmap=pos,
                        heading_rad=heading,
                        goal_pos_mmap=goal_pos,
                        path_world_mmap=self.session_.current_path_world_,
                        path_grid=self.session_.current_path_grid_,
                        target_idx=self.session_.current_target_idx_,
                        is_stuck=is_stuck,
                        path_deviation=dev,
                        distance_to_goal=dist_goal,
                        goal_reached=goal_reached,
                    )
                except Exception:
                    pass
            
            # Update data hub
            try:
                self.data_hub_.set_nav_status(
                    is_stuck=is_stuck,
                    path_deviation=dev,
                    distance_to_goal=dist_goal,
                    goal_reached=goal_reached,
                )
                
                if self.session_.current_target_idx_ != self.session_.last_published_idx_:
                    self.session_.last_published_idx_ = self.session_.current_target_idx_
                    self.data_hub_.set_current_path(
                        grid_path=self.session_.current_path_grid_,
                        world_path=self.session_.current_path_world_,
                        target_idx=self.session_.current_target_idx_,
                    )
            except Exception:
                pass
            
            # Check goal reached
            if goal_reached or target_world is None:
                self.move_.stop()
                self.session_.reset()
                time.sleep(interval)
                continue
            
            # Execute movement
            self.move_.goto(
                follow_result=follow_result,
                current_pos=pos,
                heading=heading,
            )
            
            dt = time.perf_counter() - t0
            if dt < interval:
                time.sleep(interval - dt)
        
        logger.info("Control thread exited")

    # =========================================================================
    # Legacy compatibility (deprecated, use new lifecycle methods)
    # =========================================================================
    
    def start(self, map_name: Optional[str] = None) -> bool:
        """[Deprecated] Use init() + startSession() instead."""
        if not self.isInitialized():
            if not self.init():
                return False
        return self.startSession(map_name)
    
    def stop(self) -> None:
        """[Deprecated] Use endSession() or shutdown() instead."""
        self.endSession()
    
    def is_running(self) -> bool:
        """[Deprecated] Use isRunning() instead."""
        return self.isRunning()


# =============================================================================
# Test Demo
# =============================================================================
'''
Lifecycle Demo:

1. Program startup (once):
   runtime = NavigationRuntime()
   runtime.init()                    # Load models (slow, ~2-3s)
   runtime.detectMinimapRegion()     # Detect minimap position (optional, auto-detected in startSession)

2. Each game match:
   runtime.startSession()            # Auto-detect map name, start threads
   # ... game in progress ...
   runtime.endSession()              # Stop threads, clean CUDA cache

3. Program exit (once):
   runtime.shutdown()                # Release all resources

Key points:
- Minimap region is detected once and cached (position is fixed)
- Map name is auto-detected each match (map changes each match)
- Models stay loaded across matches (no reload delay)
- CUDA cache is cleaned at endSession() to prevent memory leak

Hotkeys:
  F9: Start session (auto-detect map name)
  F10: End session  
  F11: Pause/Resume session
  ESC: Shutdown and exit
'''
if __name__ == "__main__":
    import os
    from pynput import keyboard

    logger.info(f"Current working directory: {os.getcwd()}")
    
    runtime = NavigationRuntime()
    
    def on_press(key):
        if key == keyboard.Key.f9:
            # Initialize if needed
            if runtime.getState() == NavigationRuntime.State.UNINITIALIZED:
                if not runtime.init():
                    logger.error("F9: init failed")
                    return
            
            # Start session (map name will be auto-detected)
            if runtime.getState() == NavigationRuntime.State.IDLE:
                # startSession() will auto-detect minimap region if not cached
                # and auto-detect map name if not provided
                if runtime.startSession():  # No map_name = auto-detect
                    logger.info("F9: Session started (map auto-detected)")
                else:
                    logger.error("F9: startSession failed")
            else:
                logger.info(f"F9: Already in state {runtime.getState()}")
                
        elif key == keyboard.Key.f10:
            if runtime.getState() in (NavigationRuntime.State.RUNNING, NavigationRuntime.State.PAUSED):
                runtime.endSession()
                logger.info("F10: Session ended (CUDA cleaned)")
            else:
                logger.info(f"F10: No active session (state={runtime.getState()})")
                
        elif key == keyboard.Key.f11:
            if runtime.getState() == NavigationRuntime.State.RUNNING:
                runtime.pauseSession()
                logger.info("F11: Session paused")
            elif runtime.getState() == NavigationRuntime.State.PAUSED:
                runtime.resumeSession()
                logger.info("F11: Session resumed")
            else:
                logger.info(f"F11: Cannot pause/resume in state {runtime.getState()}")
                
        elif key == keyboard.Key.esc:
            runtime.shutdown()
            logger.info("ESC: Shutdown complete")
            return False

    logger.info("F9=start, F10=end, F11=pause/resume, ESC=exit")
    
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    
    logger.info("NavigationRuntime demo exited")
