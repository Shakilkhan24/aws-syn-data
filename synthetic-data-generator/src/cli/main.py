"""Main CLI entry point."""

import sys
import argparse
from pathlib import Path
from typing import List

from ..config.loader import load_config
from ..config.validator import validate_config, ConfigValidationError
from ..core.api_client import APIClientManager
from ..core.task_executor import TaskExecutor
from ..core.batch_processor import BatchProcessor
from ..core.snapshot_tracker import SnapshotTracker
from ..utils.logger import setup_logger
import pandas as pd

logger = setup_logger(__name__)


def parse_api_keys(keys_input: str) -> List[str]:
    """
    Parse API keys from string (comma-separated or list).
    
    Args:
        keys_input: String containing API keys (comma-separated or list)
    
    Returns:
        List of API keys
    """
    if isinstance(keys_input, list):
        return keys_input
    elif isinstance(keys_input, str):
        # Handle comma-separated string
        return [key.strip() for key in keys_input.split(",") if key.strip()]
    else:
        return []


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generator - Generate synthetic data using AI APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single CSV file with default config
  python -m src.cli.main --config configs/default.yaml data/input/file.csv
  
  # Process a specific index range (rows 0-1000)
  python -m src.cli.main --config configs/default.yaml --start-index 0 --end-index 1000 data/input/file.csv
  
  # Process with checkpoints every 100 rows
  python -m src.cli.main --config configs/default.yaml --save-interval-rows 100 data/input/file.csv
  
  # Parallel processing: Process different ranges in parallel
  # Terminal 1: rows 0-1000
  python -m src.cli.main --config configs/default.yaml --start-index 0 --end-index 1000 data/input/file.csv
  # Terminal 2: rows 1000-2000
  python -m src.cli.main --config configs/default.yaml --start-index 1000 --end-index 2000 data/input/file.csv
  
  # Check status of processing
  python -m src.cli.status data/input/file.csv
  
  # Override batch size
  python -m src.cli.main --config configs/default.yaml --batch-size 50 data/input/file.csv
        """
    )
    
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Path(s) to CSV file(s) to process"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory (default: data/output)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--start-index",
        type=int,
        help="Start processing from this row index (0-based, inclusive). Useful for parallel processing."
    )
    
    parser.add_argument(
        "--end-index",
        type=int,
        help="End processing at this row index (0-based, exclusive). Useful for parallel processing."
    )
    
    parser.add_argument(
        "--save-interval-rows",
        type=int,
        help="Save checkpoint after every N rows (default: disabled). Recommended: 100"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory for checkpoint saves (default: data/checkpoints)"
    )
    
    parser.add_argument(
        "--show-status",
        action="store_true",
        help="Show snapshot status before processing"
    )
    
    args = parser.parse_args()
    
    # Setup logger with specified level
    logger = setup_logger("synthetic_data_generator", args.log_level)
    
    # Load and validate configuration
    try:
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        validate_config(config)
        logger.info("✅ Configuration validated successfully")
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except ConfigValidationError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        sys.exit(1)
    
    # Parse API keys
    api_keys = parse_api_keys(config["api"]["keys"])
    if not api_keys:
        logger.error("No API keys found in configuration")
        sys.exit(1)
    
    # Initialize API client manager
    try:
        api_client = APIClientManager(
            provider=config["api"]["provider"],
            model=config["api"]["model"],
            api_keys=api_keys,
            **config["api"].get("provider_kwargs", {})
        )
    except Exception as e:
        logger.error(f"Failed to initialize API client: {e}")
        sys.exit(1)
    
    # Initialize task executor
    try:
        task_executor = TaskExecutor(
            tasks=config["tasks"],
            api_client_manager=api_client
        )
    except Exception as e:
        logger.error(f"Failed to initialize task executor: {e}")
        sys.exit(1)
    
    # Get processing parameters
    batch_size = args.batch_size or config["processing"]["batch_size"]
    save_interval = config["processing"].get("save_interval", 1)
    retry_delay = config["processing"].get("retry_delay", 0.1)
    save_interval_rows = args.save_interval_rows or config["processing"].get("save_interval_rows")
    checkpoint_dir = args.checkpoint_dir or Path(config["processing"].get("checkpoint_dir", "data/checkpoints"))
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        api_client_manager=api_client,
        task_executor=task_executor,
        batch_size=batch_size,
        save_interval=save_interval,
        retry_delay=retry_delay,
        save_interval_rows=save_interval_rows,
        checkpoint_dir=checkpoint_dir
    )
    
    # Log index range if specified
    if args.start_index is not None or args.end_index is not None:
        logger.info(f"📊 Processing index range: {args.start_index or 0} to {args.end_index or 'end'}")
    
    # Process CSV files
    csv_files = []
    for csv_path_str in args.csv_files:
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}, skipping")
            continue
        if csv_path.suffix.lower() != ".csv":
            logger.warning(f"Not a CSV file: {csv_path}, skipping")
            continue
        csv_files.append(csv_path)
    
    if not csv_files:
        logger.error("No valid CSV files found to process")
        sys.exit(1)
    
    # Show status if requested
    if args.show_status:
        logger.info(f"\n{'='*60}")
        logger.info("CURRENT STATUS SNAPSHOT")
        logger.info(f"{'='*60}")
        for csv_file in csv_files:
            try:
                snapshot_tracker = SnapshotTracker(csv_file)
                total_rows = len(pd.read_csv(csv_file, encoding='utf-8-sig'))
                snapshot_tracker.print_snapshot(total_rows)
            except Exception as e:
                logger.warning(f"Could not show status for {csv_file.name}: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting processing of {len(csv_files)} file(s)")
    logger.info(f"{'='*60}")
    
    completed = 0
    failed = []
    
    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"\n{'─'*60}")
        logger.info(f"File {i}/{len(csv_files)}: {csv_file.name}")
        logger.info(f"{'─'*60}")
        
        try:
            success = batch_processor.process_file(
                csv_file=csv_file,
                output_dir=args.output_dir,
                start_index=args.start_index,
                end_index=args.end_index
            )
            if success:
                completed += 1
                logger.info(f"✅ Successfully completed: {csv_file.name}")
            else:
                failed.append(csv_file.name)
                logger.warning(f"⚠️ Processing incomplete: {csv_file.name}")
        except Exception as e:
            failed.append(csv_file.name)
            logger.error(f"❌ Error processing {csv_file.name}: {e}", exc_info=True)
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {len(csv_files)}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed/Incomplete: {len(failed)}")
    
    if failed:
        logger.warning(f"\nFailed/Incomplete files:")
        for file in failed:
            logger.warning(f"  - {file}")
    
    # Print API usage stats
    api_client.print_usage_stats()
    
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

