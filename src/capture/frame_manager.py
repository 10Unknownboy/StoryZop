from __future__ import annotations

from pathlib import Path
from PIL import Image

from src.config import StoryZopConfig
from src.database.models import CapturePass
from src.logger import get_logger

logger = get_logger(__name__)


class FrameManager:
    """Manages the storage and retrieval of captured story frames."""

    def __init__(self, config: StoryZopConfig):
        self.config = config

    def ensure_directories(self, person_id: str, story_id: str) -> Path:
        """Create initial and revisit directories for a story. Return the story dir."""
        story_dir = self.config.data_dir / "stories" / person_id / story_id
        (story_dir / "initial").mkdir(parents=True, exist_ok=True)
        (story_dir / "revisit").mkdir(parents=True, exist_ok=True)
        return story_dir

    def get_frame_path(
        self, person_id: str, story_id: str, capture_pass: CapturePass, frame_number: int
    ) -> Path:
        """Return a deterministic file path for a captured frame."""
        story_dir = self.ensure_directories(person_id, story_id)
        pass_dir = "initial" if capture_pass == CapturePass.INITIAL else "revisit"
        filename = f"frame_{frame_number:03d}.png"
        return story_dir / pass_dir / filename

    def get_frame_dimensions(self, frame_path: str | Path) -> tuple[int, int]:
        """Return the width and height of a frame."""
        with Image.open(frame_path) as img:
            return img.size

    def load_frame(self, frame_path: str | Path) -> Image.Image:
        """Load and return the frame image."""
        return Image.open(frame_path)

    def cleanup_temp_frames(self, directory: Path) -> None:
        """Remove any temporary files in the directory."""
        if not directory.exists():
            return
        for file_path in directory.glob("*.temp"):
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning("Failed to delete temp frame %s: %s", file_path, e)

    def get_profile_pic_path(self, person_id: str) -> Path:
        """Return the path for storing a person's profile picture."""
        pfp_dir = self.config.data_dir / "profiles" / person_id
        pfp_dir.mkdir(parents=True, exist_ok=True)
        return pfp_dir / "profile.png"
