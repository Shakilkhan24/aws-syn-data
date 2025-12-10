"""File handling utilities."""

import pandas as pd
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime


def ensure_output_columns(df: pd.DataFrame, output_columns: list) -> pd.DataFrame:
    """
    Ensure output columns exist in DataFrame, creating them if needed.
    
    Args:
        df: Input DataFrame
        output_columns: List of column names that should exist
    
    Returns:
        DataFrame with all output columns
    """
    df = df.copy()
    for col in output_columns:
        if col not in df.columns:
            df[col] = None
    return df


def get_working_file_path(original_file: Path) -> Path:
    """
    Get the path for a working copy of a file.
    
    Args:
        original_file: Path to original file
    
    Returns:
        Path to working file
    """
    return original_file.parent / f"working_{original_file.name}"


def create_working_file(original_file: Path, output_columns: list) -> Path:
    """
    Create a working copy of a CSV file with output columns added.
    
    Args:
        original_file: Path to original CSV file
        output_columns: List of output column names to add
    
    Returns:
        Path to created working file
    """
    working_file = get_working_file_path(original_file)
    
    if not working_file.exists():
        df = pd.read_csv(original_file, encoding='utf-8-sig')
        df = ensure_output_columns(df, output_columns)
        df.to_csv(working_file, index=False, encoding='utf-8-sig')
    
    return working_file


def save_dataframe(df: pd.DataFrame, file_path: Path, encoding: str = 'utf-8-sig', atomic: bool = True) -> None:
    """
    Save DataFrame to CSV with proper encoding. Uses atomic writes for safety.
    
    Args:
        df: DataFrame to save
        file_path: Destination path
        encoding: File encoding (default: utf-8-sig for Excel compatibility)
        atomic: If True, write to temp file first then rename (default: True)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if atomic:
        # Atomic write: write to temp file, then rename
        temp_file = file_path.with_suffix('.tmp')
        try:
            df.to_csv(temp_file, index=False, encoding=encoding)
            # On Windows, we need to remove target first if it exists
            if file_path.exists():
                file_path.unlink()
            temp_file.rename(file_path)
        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise e
    else:
        df.to_csv(file_path, index=False, encoding=encoding)


def save_checkpoint(df: pd.DataFrame, base_file: Path, checkpoint_dir: Path, row_count: int) -> Path:
    """
    Save a checkpoint copy of the DataFrame to a checkpoint directory.
    
    Args:
        df: DataFrame to save
        base_file: Original file path (for naming)
        checkpoint_dir: Directory to save checkpoints
        row_count: Number of rows processed (for naming)
    
    Returns:
        Path to saved checkpoint file
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint filename with timestamp and row count
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{base_file.stem}_checkpoint_r{row_count}_{timestamp}.csv"
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    save_dataframe(df, checkpoint_path, atomic=True)
    return checkpoint_path

