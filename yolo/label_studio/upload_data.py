#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload gameplay frames from C:/data/recordings to Label Studio.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Dict
import logging
import os

import yaml
from label_studio_sdk import Client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get script directory for default config path
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"


class LabelStudioUploader:
    """Upload images to Label Studio."""

    def __init__(self, url: str, api_key: str, project_name: str = "Tanks"):
        self.url_ = url
        self.api_key_ = api_key
        self.project_name_ = project_name
        self.client_ = None
        self.project_ = None

    def Connect(self):
        """Connect to Label Studio and get/create project."""
        if not self.api_key_:
            raise ValueError("API Key is required to connect to Label Studio.")

        logger.info(f"Connecting to Label Studio at {self.url_}")
        
        try:
            self.client_ = Client(url=self.url_, api_key=self.api_key_)
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize Label Studio client: {e}\n"
                f"Please check:\n"
                f"  1. Label Studio is running at {self.url_}\n"
                f"  2. API key is valid and not expired"
            ) from e

        try:
            # Test connection by getting projects
            logger.info("Testing connection...")
            projects = self.client_.get_projects()
            logger.info(f"Connection successful. Found {len(projects) if projects else 0} existing projects")
            
            existing_project = None
            if projects:
                for proj in projects:
                    title = getattr(proj, 'title', None) or getattr(proj, 'name', None)
                    if title == self.project_name_:
                        existing_project = proj
                        break

            if existing_project:
                project_id = getattr(existing_project, 'id', None)
                logger.info(f"Using existing project: {self.project_name_} (ID: {project_id})")
                self.project_ = self.client_.get_project(project_id)
            else:
                logger.info(f"Creating new project: {self.project_name_}")
                self.project_ = self.client_.start_project(title=self.project_name_)
                
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                raise ConnectionError(
                    f"Authentication failed (401 Unauthorized):\n"
                    f"  - The API key may be invalid or expired\n"
                    f"  - Please check your API key in Label Studio settings\n"
                    f"  - You can get a new API key from: {self.url_}/user/account"
                ) from e
            elif "Connection" in error_msg or "refused" in error_msg.lower():
                raise ConnectionError(
                    f"Cannot connect to Label Studio at {self.url_}\n"
                    f"  - Please ensure Label Studio is running\n"
                    f"  - Check if the URL is correct"
                ) from e
            else:
                raise ConnectionError(f"Failed to connect or create project: {e}") from e

    def CollectImagesFromRecordings(
        self,
        recordings_dir: str,
        max_sessions: Optional[int] = None,
        max_frames_per_session: Optional[int] = None
    ) -> List[dict]:
        """Collect images from recording sessions."""
        recordings_path = Path(recordings_dir)
        if not recordings_path.exists():
            raise ValueError(f"Recordings directory does not exist: {recordings_dir}")

        session_dirs = sorted([d for d in recordings_path.iterdir()
                               if d.is_dir() and d.name.startswith("session_")])
        if max_sessions:
            session_dirs = session_dirs[:max_sessions]

        logger.info(f"Found {len(session_dirs)} recording sessions")

        tasks = []
        for session_idx, session_dir in enumerate(session_dirs):
            logger.info(f"Processing session {session_idx + 1}/{len(session_dirs)}: {session_dir.name}")

            frames_dir = session_dir / "frames"
            if not frames_dir.exists():
                logger.warning(f"Frames directory not found: {frames_dir}")
                continue

            image_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
            if max_frames_per_session:
                image_files = image_files[:max_frames_per_session]

            logger.info(f"  Found {len(image_files)} images in {session_dir.name}")

            for image_file in image_files:
                image_path = str(image_file.absolute()).replace('\\', '/')
                tasks.append({
                    "data": {
                        "image": f"file://{image_path}" if not image_path.startswith('/') else image_path
                    }
                })

        logger.info(f"Total images collected: {len(tasks)}")
        return tasks

    def UploadTasks(self, tasks: List[dict], batch_size: int = 50):
        """Upload tasks to Label Studio in batches."""
        if self.project_ is None:
            raise RuntimeError("Not connected. Call Connect() first.")

        total_batches = (len(tasks) + batch_size - 1) // batch_size
        logger.info(f"Uploading {len(tasks)} tasks in {total_batches} batches (batch size: {batch_size})")

        for i in range(0, len(tasks), batch_size):
            batch_num = (i // batch_size) + 1
            batch = tasks[i:i + batch_size]

            try:
                logger.info(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} tasks)")
                self.project_.import_tasks(batch)
                logger.info(f"Batch {batch_num} uploaded successfully")
            except Exception as e:
                logger.error(f"Error uploading batch {batch_num}: {e}")
                continue

        logger.info("All batches uploaded")


def LoadConfig(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
    
    Returns:
        Dictionary with configuration values.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Extract label_studio section
        if config and 'label_studio' in config:
            return config['label_studio']
        return {}
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Upload gameplay frames to Label Studio")
    parser.add_argument("--recordings-dir", type=str, default=r"C:/data/recordings",
                        help="Directory containing recording sessions")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file (default: yolo/label_studio/config.yaml)")
    parser.add_argument("--url", type=str, default=None,
                        help="Label Studio server URL (overrides config)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Label Studio API key (overrides config/env)")
    parser.add_argument("--project-name", type=str, default=None,
                        help="Label Studio project name (overrides config)")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Maximum number of sessions to process")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames per session to process")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for upload")
    parser.add_argument("--test-mode", action="store_true",
                        help="Test mode: process only first session with limited frames")
    args = parser.parse_args()
    
    # Load config from YAML file
    config = LoadConfig(args.config)
    
    # Priority: command line args > config file > environment variables > defaults
    url = args.url or config.get('url') or "http://localhost:8080"
    api_key = args.api_key or config.get('api_key') or os.getenv("LABEL_STUDIO_API_KEY")
    project_name = args.project_name or config.get('project_name') or "Tank Images Upload"

    if args.test_mode:
        args.max_sessions = 1
        args.max_frames = 10
        logger.info("Running in test mode")

    if not api_key:
        logger.error("API Key is required!")
        logger.error("  Option 1: Set in config.yaml file")
        logger.error("  Option 2: Set environment variable: set LABEL_STUDIO_API_KEY=your_api_key")
        logger.error("  Option 3: Use command line: --api-key your_api_key")
        logger.error(f"\n  You can get your API key from: {url}/user/account")
        raise ValueError("API Key is required!")

    logger.info(f"Using Label Studio configuration:")
    logger.info(f"  URL: {url}")
    logger.info(f"  Project: {project_name}")
    logger.info(f"  API Key: {'*' * (len(api_key) - 8) + api_key[-8:] if api_key else 'Not set'}")

    uploader = LabelStudioUploader(
        url=url,
        api_key=api_key,
        project_name=project_name
    )

    try:
        uploader.Connect()
        tasks = uploader.CollectImagesFromRecordings(
            recordings_dir=args.recordings_dir,
            max_sessions=args.max_sessions,
            max_frames_per_session=args.max_frames
        )

        if not tasks:
            logger.warning("No images found to upload")
            return

        uploader.UploadTasks(tasks, batch_size=args.batch_size)
        logger.info("\nUpload completed successfully!")

    except ConnectionError as e:
        logger.error(f"\nConnection Error:")
        logger.error(str(e))
        raise
    except ValueError as e:
        logger.error(f"\nValue Error: {e}")
        raise
    except Exception as e:
        logger.error(f"\nUnexpected error during upload: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
