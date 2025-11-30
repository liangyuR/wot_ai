import os
import time
import psutil
from loguru import logger


class GameRestarter:
    """
    Game restarter class to handle game termination and relaunch.
    Follows Google C++ style guide for naming conventions as requested.
    """

    def __init__(self, process_name: str, game_exe_path: str, wait_seconds: int = 10):
        """
        Initializes the GameRestarter.

        Args:
            process_name: The name of the process to kill (e.g., "WorldOfTanks.exe").
            game_exe_path: The full path to the game executable.
            wait_seconds: Time to wait between killing and starting the game.
        """
        self.process_name_ = process_name
        self.game_exe_path_ = game_exe_path
        self.wait_seconds_ = wait_seconds

    def KillGame(self) -> None:
        """
        Forcefully terminates the game process.
        """
        logger.info(f"Attempting to kill process: {self.process_name_}")
        self.killProcessByName_()

    def StartGame(self) -> None:
        """
        Starts the game executable independently.
        """
        logger.info(f"Attempting to start game from: {self.game_exe_path_}")
        if not os.path.exists(self.game_exe_path_):
            logger.error(f"Game executable not found at: {self.game_exe_path_}")
            return

        try:
            # os.startfile is Windows specific and launches the file as if double-clicked.
            # This creates a completely independent process.
            os.startfile(self.game_exe_path_)
            logger.info("Game start command executed.")
        except Exception as e:
            logger.error(f"Failed to start game: {e}")

    def RestartGame(self) -> None:
        """
        Restarts the game: Kills it, waits, then starts it.
        """
        logger.info("Initiating game restart sequence...")
        self.KillGame()
        
        logger.info(f"Waiting for {self.wait_seconds_} seconds before starting...")
        time.sleep(self.wait_seconds_)
        
        self.StartGame()
        logger.info("Game restart sequence completed.")

    def killProcessByName_(self) -> None:
        """
        Internal method to find and kill processes by name.
        """
        terminated_count = 0
        # Iterate over all running processes
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # Check if process name matches
                if proc.info['name'] and proc.info['name'].lower() == self.process_name_.lower():
                    pid = proc.info['pid']
                    logger.info(f"Found process {self.process_name_} (PID: {pid}), terminating...")
                    p = psutil.Process(pid)
                    p.terminate()  # Try terminate first
                    try:
                        p.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process {pid} did not terminate, forcing kill...")
                        p.kill()
                    terminated_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logger.warning(f"Error processing process: {e}")

        if terminated_count > 0:
            logger.info(f"Successfully terminated {terminated_count} instance(s) of {self.process_name_}.")
        else:
            logger.info(f"No running instances of {self.process_name_} found.")

