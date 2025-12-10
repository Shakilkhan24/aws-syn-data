"""Batch processing logic for CSV files."""

import time
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from ..utils.logger import setup_logger
from ..utils.file_handler import (
    create_working_file, 
    save_dataframe, 
    get_working_file_path,
    save_checkpoint
)
from ..core.progress_tracker import ProgressTracker
from ..core.snapshot_tracker import SnapshotTracker
from ..core.task_executor import TaskExecutor
from ..core.api_client import APIClientManager

logger = setup_logger(__name__)


class BatchProcessor:
    """Processes CSV files in batches with progress tracking."""
    
    def __init__(
        self,
        api_client_manager: APIClientManager,
        task_executor: TaskExecutor,
        batch_size: int = 100,
        save_interval: int = 1,
        retry_delay: float = 0.1,
        save_interval_rows: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            api_client_manager: API client manager
            task_executor: Task executor
            batch_size: Number of rows per batch
            save_interval: Save after every N batches
            retry_delay: Delay between API requests (seconds)
            save_interval_rows: Save checkpoint after every N rows (default: None, disabled)
            checkpoint_dir: Directory for checkpoint saves (default: data/checkpoints)
        """
        self.api_client = api_client_manager
        self.task_executor = task_executor
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.retry_delay = retry_delay
        self.save_interval_rows = save_interval_rows
        self.checkpoint_dir = checkpoint_dir or Path("data/checkpoints")
    
    def process_file(
        self,
        csv_file: Path,
        output_dir: Optional[Path] = None,
        progress_dir: Optional[Path] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None
    ) -> bool:
        """
        Process a single CSV file, optionally for a specific index range.
        
        Args:
            csv_file: Path to input CSV file
            output_dir: Directory for output files (default: data/output)
            progress_dir: Directory for progress files (default: data/progress)
            start_index: Starting row index (0-based, inclusive). If None, starts from beginning.
            end_index: Ending row index (0-based, exclusive). If None, processes to end.
        
        Returns:
            True if processing completed successfully, False otherwise
        """
        if output_dir is None:
            output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracker with range info
        progress_tracker = ProgressTracker(csv_file, progress_dir, start_index, end_index)
        
        # Initialize snapshot tracker
        snapshot_tracker = SnapshotTracker(csv_file)
        
        # Create working file
        output_columns = self.task_executor.get_output_columns()
        working_file = create_working_file(csv_file, output_columns)
        
        logger.info(f"Processing: {csv_file.name}")
        logger.info(f"Working file: {working_file.name}")
        logger.info(f"Progress file: {progress_tracker.progress_file.name}")
        
        # Load DataFrame
        try:
            df = pd.read_csv(working_file, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return False
        
        total_rows = len(df)
        
        # Apply index range if specified
        if start_index is not None or end_index is not None:
            start_idx = start_index if start_index is not None else 0
            end_idx = end_index if end_index is not None else total_rows
            
            # Validate range
            if start_idx < 0:
                logger.error(f"Invalid start_index: {start_idx}. Must be >= 0")
                return False
            if end_idx > total_rows:
                logger.warning(f"end_index {end_idx} exceeds total rows {total_rows}, using {total_rows}")
                end_idx = total_rows
            if start_idx >= end_idx:
                logger.error(f"Invalid range: start_index {start_idx} >= end_index {end_idx}")
                return False
            
            range_rows = end_idx - start_idx
            logger.info(f"Processing range: rows {start_idx} to {end_idx-1} ({range_rows:,} rows)")
        else:
            start_idx = 0
            end_idx = total_rows
            range_rows = total_rows
            logger.info(f"Processing entire file: {total_rows:,} rows")
        
        # Keep reference to full DataFrame for updates
        # We'll work on the full df but only process the specified range
        
        # Get resume point (relative to range start)
        current_batch, completed_rows = progress_tracker.get_resume_point()
        
        if completed_rows > 0:
            logger.info(f"Resuming from batch {current_batch + 1}, row {completed_rows} (absolute: {start_idx + completed_rows})")
        else:
            logger.info("Starting fresh processing")
        
        # Calculate batches for the range
        total_batches = (range_rows + self.batch_size - 1) // self.batch_size
        start_row = completed_rows
        
        # Track last checkpoint row count
        last_checkpoint_rows = 0
        
        try:
            for batch_num in range(current_batch, total_batches):
                batch_start = batch_num * self.batch_size
                batch_end = min(batch_start + self.batch_size, total_rows)
                
                # Adjust start if resuming mid-batch
                if batch_num == current_batch:
                    batch_start = max(batch_start, start_row)
                
                logger.info(f"\nProcessing Batch {batch_num + 1}/{total_batches} "
                          f"(rows {batch_start + 1} to {batch_end})")
                
                key_info = self.api_client.get_current_key_info()
                logger.info(f"Active API Key: {key_info['name']} | "
                          f"Usage: {key_info['usage_count']} requests")
                
                # Process batch with progress bar
                rows_to_process = range(batch_start, batch_end)
                with tqdm(rows_to_process, desc=f"Batch {batch_num + 1}") as pbar:
                    for relative_idx in pbar:
                        # Calculate absolute index in full DataFrame
                        absolute_idx = start_idx + relative_idx
                        
                        # Get row from full DataFrame
                        row = df.iloc[absolute_idx]
                        results = self.task_executor.process_row(row, absolute_idx)
                        
                        # Update full DataFrame at absolute index
                        for output_col, value in results.items():
                            df.at[absolute_idx, output_col] = value
                        
                        # Calculate relative row for progress tracking
                        relative_row = relative_idx + 1
                        
                        # Update progress (use relative row for range processing)
                        progress_tracker.save_progress(batch_num, relative_row, range_rows)
                        
                        # Save checkpoint if interval reached
                        if self.save_interval_rows and relative_row - last_checkpoint_rows >= self.save_interval_rows:
                            checkpoint_path = save_checkpoint(df, csv_file, self.checkpoint_dir, absolute_idx)
                            logger.info(f"💾 Checkpoint saved: {checkpoint_path.name} (row {absolute_idx})")
                            last_checkpoint_rows = relative_row
                        
                        # Delay to avoid overwhelming API
                        time.sleep(self.retry_delay)
                
                # Save after each batch (or according to save_interval)
                if (batch_num + 1) % self.save_interval == 0:
                    save_dataframe(df, working_file, atomic=True)
                    logger.info(f"💾 Saved progress after batch {batch_num + 1}")
                
                # Save final progress for batch
                progress_tracker.save_progress(batch_num + 1, batch_end, range_rows)
                
                logger.info(f"✅ Batch {batch_num + 1} completed")
            
            # Processing completed successfully
            # Save the full working file (with all updates)
            save_dataframe(df, working_file, atomic=True)
            logger.info(f"💾 Working file updated: {working_file.name}")
            
            # If processing a range, save range-specific output
            if start_index is not None or end_index is not None:
                range_suffix = f"_r{start_idx}-{end_idx-1}"
                final_file = output_dir / f"final_{csv_file.stem}{range_suffix}.csv"
                # Save only the range subset to the final file
                range_df = df.iloc[start_idx:end_idx].copy()
                save_dataframe(range_df, final_file, atomic=True)
                logger.info(f"\n🎉 Processing completed! Range output: {final_file}")
                
                # Mark range as completed in snapshot
                snapshot_tracker.mark_range_completed(start_idx, end_idx, total_rows)
            else:
                final_file = output_dir / f"final_{csv_file.stem}.csv"
                save_dataframe(df, final_file, atomic=True)
                logger.info(f"\n🎉 Processing completed! Final output: {final_file}")
                
                # Mark entire file as completed
                snapshot_tracker.mark_range_completed(0, total_rows, total_rows)
            
            # Save final checkpoint
            if self.save_interval_rows:
                final_checkpoint = save_checkpoint(df, csv_file, self.checkpoint_dir, end_idx - 1)
                logger.info(f"💾 Final checkpoint saved: {final_checkpoint.name}")
            
            # Print snapshot status
            snapshot_tracker.print_snapshot(total_rows)
            
            # Clean up progress file
            progress_tracker.cleanup()
            
            # Optionally remove working file
            if working_file.exists():
                working_file.unlink()
                logger.info(f"Removed working file: {working_file.name}")
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("\n⏸️ Processing interrupted by user")
            save_dataframe(df, working_file, atomic=True)
            logger.info(f"💾 Progress saved to: {working_file}")
            # Save emergency checkpoint
            if self.save_interval_rows:
                try:
                    emergency_checkpoint = save_checkpoint(df, csv_file, self.checkpoint_dir, start_idx + completed_rows)
                    logger.info(f"💾 Emergency checkpoint saved: {emergency_checkpoint.name}")
                except Exception as e:
                    logger.warning(f"Failed to save emergency checkpoint: {e}")
            return False
        
        except Exception as e:
            logger.error(f"\n❌ Error during processing: {e}", exc_info=True)
            save_dataframe(df, working_file, atomic=True)
            logger.info(f"💾 Progress saved to: {working_file}")
            # Save emergency checkpoint
            if self.save_interval_rows:
                try:
                    emergency_checkpoint = save_checkpoint(df, csv_file, self.checkpoint_dir, start_idx + completed_rows)
                    logger.info(f"💾 Emergency checkpoint saved: {emergency_checkpoint.name}")
                except Exception as e:
                    logger.warning(f"Failed to save emergency checkpoint: {e}")
            return False

