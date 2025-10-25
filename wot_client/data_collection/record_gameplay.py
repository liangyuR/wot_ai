"""
Record gameplay data for imitation learning
"""

import argparse
import cv2
import numpy as np
import time
import json
from pathlib import Path
from pynput import keyboard, mouse
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent))

try:
    from cpp_bindings import ScreenCapture
    CPP_AVAILABLE = True
except ImportError:
    logger.warning("C++ bindings not available, using fallback")
    CPP_AVAILABLE = False
    import mss


class GameplayRecorder:
    """Record gameplay footage and input data"""
    
    def __init__(self, output_dir: str = "data/recordings", fps: int = 30):
        """
        Initialize recorder
        
        Args:
            output_dir: Output directory for recordings
            fps: Recording FPS
        """
        self.output_dir_ = Path(output_dir)
        self.output_dir_.mkdir(parents=True, exist_ok=True)
        
        self.fps_ = fps
        self.frame_interval_ = 1.0 / fps
        
        # Recording state
        self.recording_ = False
        self.frames_ = []
        self.actions_ = []
        self.timestamps_ = []
        
        # Input tracking
        self.current_keys_ = set()
        self.mouse_position_ = (0, 0)
        
        # Initialize screen capture
        if CPP_AVAILABLE:
            self.screen_capture_ = ScreenCapture(1920, 1080)
        else:
            self.sct_ = mss.mss()
            self.monitor_ = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        
        # Setup input listeners
        self.setupInputListeners()
        
    def setupInputListeners(self):
        """Setup keyboard and mouse listeners"""
        
        # Keyboard listener
        self.keyboard_listener_ = keyboard.Listener(
            on_press=self.onKeyPress,
            on_release=self.onKeyRelease
        )
        
        # Mouse listener
        self.mouse_listener_ = mouse.Listener(
            on_move=self.onMouseMove,
            on_click=self.onMouseClick
        )
        
        self.keyboard_listener_.start()
        self.mouse_listener_.start()
        
    def onKeyPress(self, key):
        """Keyboard press callback"""
        try:
            self.current_keys_.add(key.char)
        except AttributeError:
            # Special key
            self.current_keys_.add(str(key))
            
    def onKeyRelease(self, key):
        """Keyboard release callback"""
        try:
            self.current_keys_.discard(key.char)
        except AttributeError:
            self.current_keys_.discard(str(key))
            
        # Stop recording on ESC
        if key == keyboard.Key.esc:
            self.stopRecording()
            
    def onMouseMove(self, x, y):
        """Mouse move callback"""
        self.mouse_position_ = (x, y)
        
    def onMouseClick(self, x, y, button, pressed):
        """Mouse click callback"""
        if pressed:
            self.current_keys_.add(f"mouse_{button.name}")
        else:
            self.current_keys_.discard(f"mouse_{button.name}")
            
    def captureScreen(self) -> np.ndarray:
        """Capture current screen"""
        if CPP_AVAILABLE:
            buffer = self.screen_capture_.Capture()
            # Convert to numpy array and reshape
            frame = np.frombuffer(buffer, dtype=np.uint8)
            frame = frame.reshape((1080, 1920, 3))
        else:
            screenshot = self.sct_.grab(self.monitor_)
            frame = np.array(screenshot)[:, :, :3]  # Remove alpha channel
            
        return frame
        
    def getCurrentAction(self) -> dict:
        """Get current input action"""
        return {
            "keys": list(self.current_keys_),
            "mouse_pos": self.mouse_position_
        }
        
    def startRecording(self):
        """Start recording"""
        self.recording_ = True
        self.frames_ = []
        self.actions_ = []
        self.timestamps_ = []
        
        logger.info("Recording started! Press ESC to stop.")
        
        start_time = time.time()
        
        try:
            while self.recording_:
                frame_start = time.time()
                
                # Capture frame
                frame = self.captureScreen()
                action = self.getCurrentAction()
                timestamp = time.time() - start_time
                
                # Store data
                self.frames_.append(frame)
                self.actions_.append(action)
                self.timestamps_.append(timestamp)
                
                # Maintain FPS
                elapsed = time.time() - frame_start
                sleep_time = max(0, self.frame_interval_ - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Recording interrupted")
            
        self.stopRecording()
        
    def stopRecording(self):
        """Stop recording and save data"""
        if not self.recording_:
            return
            
        self.recording_ = False
        logger.info("Stopping recording...")
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        recording_dir = self.output_dir_ / f"recording_{timestamp}"
        recording_dir.mkdir(parents=True, exist_ok=True)
        
        # Save video
        logger.info("Saving video...")
        video_path = recording_dir / "gameplay.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.fps_,
            (1920, 1080)
        )
        
        for frame in self.frames_:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
        out.release()
        
        # Save actions
        logger.info("Saving action data...")
        actions_path = recording_dir / "actions.json"
        with open(actions_path, 'w') as f:
            json.dump({
                "fps": self.fps_,
                "num_frames": len(self.frames_),
                "timestamps": self.timestamps_,
                "actions": self.actions_
            }, f, indent=2)
            
        logger.info(f"Recording saved to {recording_dir}")
        logger.info(f"Total frames: {len(self.frames_)}")
        logger.info(f"Duration: {self.timestamps_[-1]:.2f}s")
        
    def cleanup(self):
        """Clean up resources"""
        self.keyboard_listener_.stop()
        self.mouse_listener_.stop()


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Record World of Tanks gameplay")
    parser.add_argument(
        "--output",
        type=str,
        default="data/recordings",
        help="Output directory"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording FPS"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Gameplay Recorder")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Recording FPS: {args.fps}")
    logger.info("")
    logger.info("Instructions:")
    logger.info("1. Start World of Tanks and enter a battle")
    logger.info("2. Press any key to start recording")
    logger.info("3. Play normally")
    logger.info("4. Press ESC to stop recording")
    logger.info("=" * 80)
    
    input("Press Enter to continue...")
    
    # Create recorder
    recorder = GameplayRecorder(args.output, args.fps)
    
    try:
        recorder.startRecording()
    finally:
        recorder.cleanup()


if __name__ == "__main__":
    main()

