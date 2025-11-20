"""
Path management for datasets, outputs, and checkpoints.
"""
from pathlib import Path


class ProjectPaths:
    """
    Centralized path management for the project.
    """

    def __init__(self, project_root: str | Path | None = None):
        """
        Initialize project paths.

        Args:
            project_root: Root directory of the project. If None, uses current file location.
        """
        if project_root is None:
            # Assume this file is in src/common/, so project root is 2 levels up
            project_root = Path(__file__).parent.parent.parent

        self.root = Path(project_root).resolve()

        # Data directories
        self.data_root = self.root / "data"
        self.raw_data = self.data_root / "raw"
        self.processed_data = self.data_root / "processed"

        # Output directories
        self.outputs = self.root / "outputs"
        self.checkpoints = self.outputs / "checkpoints"
        self.logs = self.outputs / "logs"
        self.visualizations = self.outputs / "visualizations"

        # Config directory
        self.configs = self.root / "configs"

    def create_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_root,
            self.raw_data,
            self.processed_data,
            self.outputs,
            self.checkpoints,
            self.logs,
            self.visualizations,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get path to a specific dataset in raw data."""
        return self.raw_data / dataset_name

    def get_split_path(self, split: str) -> Path:
        """Get path to a processed split (train/val/test)."""
        return self.processed_data / f"{split}.json"

    def get_checkpoint_path(self, name: str) -> Path:
        """Get path to a checkpoint file."""
        return self.checkpoints / name

    def get_log_path(self, name: str) -> Path:
        """Get path to a log file."""
        return self.logs / name


# Global instance for convenience
default_paths = ProjectPaths()
