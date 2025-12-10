"""Snapshot tracker for monitoring completed row ranges."""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SnapshotTracker:
    """Tracks completed row ranges for easy monitoring."""
    
    def __init__(self, csv_filename: Path, snapshot_dir: Optional[Path] = None):
        """
        Initialize snapshot tracker.
        
        Args:
            csv_filename: Path to the CSV file being processed
            snapshot_dir: Directory to store snapshot files (default: data/snapshots)
        """
        self.csv_filename = csv_filename
        self.csv_stem = csv_filename.stem
        
        if snapshot_dir is None:
            snapshot_dir = Path("data/snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        self.snapshot_file = snapshot_dir / f"{self.csv_stem}_snapshot.json"
        self.snapshot_data = self.load_snapshot()
    
    def load_snapshot(self) -> Dict:
        """
        Load snapshot data from file.
        
        Returns:
            Snapshot data dictionary
        """
        try:
            if self.snapshot_file.exists():
                with open(self.snapshot_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded snapshot from {self.snapshot_file}")
                    return data
        except Exception as e:
            logger.warning(f"Error loading snapshot: {e}")
        
        return {
            "total_rows": 0,
            "completed_ranges": [],
            "last_updated": None
        }
    
    def save_snapshot(self, total_rows: int, completed_ranges: List[Tuple[int, int]]) -> None:
        """
        Save snapshot data to file.
        
        Args:
            total_rows: Total number of rows in the file
            completed_ranges: List of (start, end) tuples for completed ranges
        """
        self.snapshot_data = {
            "total_rows": total_rows,
            "completed_ranges": completed_ranges,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(self.snapshot_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def mark_range_completed(self, start_index: int, end_index: int, total_rows: int) -> None:
        """
        Mark a range as completed.
        
        Args:
            start_index: Starting row index (inclusive)
            end_index: Ending row index (exclusive)
            total_rows: Total number of rows in the file
        """
        # Load current snapshot
        snapshot = self.load_snapshot()
        completed_ranges = snapshot.get("completed_ranges", [])
        
        # Add new range (merge if overlapping)
        new_range = (start_index, end_index)
        completed_ranges = self._merge_ranges(completed_ranges + [new_range])
        
        # Save updated snapshot
        self.save_snapshot(total_rows, completed_ranges)
        logger.info(f"✅ Marked range {start_index}-{end_index-1} as completed")
    
    def _merge_ranges(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge overlapping or adjacent ranges.
        
        Args:
            ranges: List of (start, end) tuples
        
        Returns:
            Merged list of non-overlapping ranges
        """
        if not ranges:
            return []
        
        # Sort by start index
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        merged = [sorted_ranges[0]]
        
        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            
            # If current range overlaps or is adjacent to last range, merge them
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        return merged
    
    def get_completed_ranges(self) -> List[Tuple[int, int]]:
        """
        Get list of completed ranges.
        
        Returns:
            List of (start, end) tuples
        """
        return self.snapshot_data.get("completed_ranges", [])
    
    def get_pending_ranges(self, total_rows: int) -> List[Tuple[int, int]]:
        """
        Get list of pending (not completed) ranges.
        
        Args:
            total_rows: Total number of rows in the file
        
        Returns:
            List of (start, end) tuples for pending ranges
        """
        completed = self.get_completed_ranges()
        
        if not completed:
            return [(0, total_rows)]
        
        pending = []
        last_end = 0
        
        for start, end in completed:
            if start > last_end:
                pending.append((last_end, start))
            last_end = max(last_end, end)
        
        if last_end < total_rows:
            pending.append((last_end, total_rows))
        
        return pending
    
    def get_completion_stats(self, total_rows: int) -> Dict:
        """
        Get completion statistics.
        
        Args:
            total_rows: Total number of rows in the file
        
        Returns:
            Dictionary with completion statistics
        """
        completed_ranges = self.get_completed_ranges()
        pending_ranges = self.get_pending_ranges(total_rows)
        
        completed_count = sum(end - start for start, end in completed_ranges)
        pending_count = sum(end - start for start, end in pending_ranges)
        
        completion_percentage = (completed_count / total_rows * 100) if total_rows > 0 else 0
        
        return {
            "total_rows": total_rows,
            "completed_rows": completed_count,
            "pending_rows": pending_count,
            "completion_percentage": round(completion_percentage, 2),
            "completed_ranges": completed_ranges,
            "pending_ranges": pending_ranges,
            "num_completed_ranges": len(completed_ranges),
            "num_pending_ranges": len(pending_ranges)
        }
    
    def print_snapshot(self, total_rows: int) -> None:
        """
        Print a visual snapshot of completion status.
        
        Args:
            total_rows: Total number of rows in the file
        """
        stats = self.get_completion_stats(total_rows)
        
        print("\n" + "=" * 70)
        print(f"📊 SNAPSHOT: {self.csv_stem}")
        print("=" * 70)
        print(f"Total Rows: {stats['total_rows']:,}")
        print(f"Completed: {stats['completed_rows']:,} ({stats['completion_percentage']}%)")
        print(f"Pending: {stats['pending_rows']:,} ({100 - stats['completion_percentage']:.2f}%)")
        print(f"\n✅ Completed Ranges ({stats['num_completed_ranges']}):")
        
        if stats['completed_ranges']:
            for start, end in stats['completed_ranges']:
                print(f"   [{start:,} - {end-1:,}] ({end-start:,} rows)")
        else:
            print("   None")
        
        print(f"\n⏳ Pending Ranges ({stats['num_pending_ranges']}):")
        if stats['pending_ranges']:
            for start, end in stats['pending_ranges']:
                print(f"   [{start:,} - {end-1:,}] ({end-start:,} rows)")
        else:
            print("   None - All done! 🎉")
        
        # Visual representation
        if total_rows > 0:
            print(f"\n📈 Visual Progress:")
            self._print_visual_progress(stats, total_rows)
        
        if self.snapshot_data.get("last_updated"):
            print(f"\nLast Updated: {self.snapshot_data['last_updated']}")
        print("=" * 70 + "\n")
    
    def _print_visual_progress(self, stats: Dict, total_rows: int, width: int = 50) -> None:
        """
        Print a visual progress bar showing completed vs pending ranges.
        
        Args:
            stats: Completion statistics
            total_rows: Total number of rows
            width: Width of the progress bar in characters
        """
        completed_ranges = stats['completed_ranges']
        pending_ranges = stats['pending_ranges']
        
        # Create a simple visual representation
        bar = [' '] * width
        scale = total_rows / width
        
        # Mark completed ranges
        for start, end in completed_ranges:
            start_pos = int(start / scale)
            end_pos = int(end / scale)
            for i in range(start_pos, min(end_pos, width)):
                bar[i] = '█'
        
        # Mark pending ranges
        for start, end in pending_ranges:
            start_pos = int(start / scale)
            end_pos = int(end / scale)
            for i in range(start_pos, min(end_pos, width)):
                if bar[i] == ' ':
                    bar[i] = '░'
        
        print(f"   [{''.join(bar)}]")
        print(f"   0{' ' * (width//2 - 1)}{total_rows//2:,}{' ' * (width//2 - len(str(total_rows)) - 1)}{total_rows:,}")

