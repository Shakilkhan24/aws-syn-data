"""CLI command to show snapshot status of CSV files."""

import sys
import argparse
from pathlib import Path
from ..core.snapshot_tracker import SnapshotTracker
from ..utils.logger import setup_logger
import pandas as pd

logger = setup_logger(__name__)


def get_file_row_count(csv_file: Path) -> int:
    """
    Get total row count from CSV file.
    
    Args:
        csv_file: Path to CSV file
    
    Returns:
        Number of rows
    """
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        return len(df)
    except Exception as e:
        logger.warning(f"Could not read {csv_file}: {e}")
        return 0


def main():
    """Show snapshot status for CSV files."""
    parser = argparse.ArgumentParser(
        description="Show snapshot status of CSV processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show status for a single file
  python -m src.cli.status data/input/file.csv
  
  # Show status for multiple files
  python -m src.cli.status data/input/*.csv
  
  # Show status for all files in directory
  python -m src.cli.status data/input/
        """
    )
    
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Path(s) to CSV file(s) or directory to check"
    )
    
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help="Directory containing snapshot files (default: data/snapshots)"
    )
    
    args = parser.parse_args()
    
    # Collect CSV files
    csv_files = []
    for path_str in args.csv_files:
        path = Path(path_str)
        if path.is_dir():
            # Add all CSV files in directory
            csv_files.extend(path.glob("*.csv"))
        elif path.exists() and path.suffix.lower() == '.csv':
            csv_files.append(path)
        else:
            logger.warning(f"Skipping invalid path: {path}")
    
    if not csv_files:
        logger.error("No valid CSV files found")
        sys.exit(1)
    
    # Show status for each file
    for csv_file in csv_files:
        snapshot_tracker = SnapshotTracker(csv_file, args.snapshot_dir)
        total_rows = get_file_row_count(csv_file)
        
        if total_rows > 0:
            snapshot_tracker.print_snapshot(total_rows)
        else:
            logger.warning(f"Could not determine row count for {csv_file.name}")


if __name__ == "__main__":
    main()

