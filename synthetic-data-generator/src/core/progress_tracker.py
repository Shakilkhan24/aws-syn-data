"""Progress tracking and resumability."""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ProgressTracker:
    """Tracks processing progress and enables resumability."""
    
    def __init__(
        self, 
        csv_filename: Path, 
        progress_dir: Optional[Path] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            csv_filename: Path to the CSV file being processed
            progress_dir: Directory to store progress files (default: data/progress)
            start_index: Starting row index (for range processing)
            end_index: Ending row index (for range processing)
        """
        self.csv_filename = csv_filename
        self.csv_stem = csv_filename.stem
        self.start_index = start_index
        self.end_index = end_index
        
        if progress_dir is None:
            progress_dir = Path("data/progress")
        progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Include range info in progress filename if processing a range
        if start_index is not None or end_index is not None:
            range_str = f"_r{start_index or 0}-{end_index or 'end'}"
            self.progress_file = progress_dir / f"{self.csv_stem}{range_str}_progress.json"
        else:
            self.progress_file = progress_dir / f"{self.csv_stem}_progress.json"
        
        self.progress_data = self.load_progress()
    
    def load_progress(self) -> Dict:
        """
        Load progress data from file.
        
        Returns:
            Progress data dictionary
        """
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Loaded progress from {self.progress_file}")
                    return data
        except Exception as e:
            logger.warning(f"Error loading progress: {e}")
        
        return {
            "current_batch": 0,
            "completed_rows": 0,
            "total_rows": 0,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "last_updated": None
        }
    
    def save_progress(self, current_batch: int, completed_rows: int, total_rows: int) -> None:
        """
        Save current progress to file.
        
        Args:
            current_batch: Current batch number (0-indexed)
            completed_rows: Number of completed rows
            total_rows: Total number of rows
        """
        from datetime import datetime
        
        self.progress_data = {
            "current_batch": current_batch,
            "completed_rows": completed_rows,
            "total_rows": total_rows,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def get_resume_point(self) -> tuple[int, int]:
        """
        Get the point from which to resume processing.
        
        Returns:
            Tuple of (current_batch, completed_rows)
        """
        return (
            self.progress_data.get("current_batch", 0),
            self.progress_data.get("completed_rows", 0)
        )
    
    def cleanup(self) -> None:
        """Remove progress file after successful completion."""
        try:
            if self.progress_file.exists():
                os.remove(self.progress_file)
                logger.info(f"Cleaned up progress file: {self.progress_file}")
        except Exception as e:
            logger.warning(f"Error cleaning up progress file: {e}")

