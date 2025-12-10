"""Task execution engine."""

import pandas as pd
from typing import List, Dict, Any, Optional
from ..utils.logger import setup_logger
from ..core.api_client import APIClientManager

logger = setup_logger(__name__)


class TaskExecutor:
    """Executes tasks on DataFrame rows."""
    
    def __init__(self, tasks: List[Dict[str, Any]], api_client_manager: APIClientManager):
        """
        Initialize task executor.
        
        Args:
            tasks: List of task configurations
            api_client_manager: API client manager instance
        """
        self.tasks = tasks
        self.api_client = api_client_manager
    
    def get_output_columns(self) -> List[str]:
        """
        Get list of output column names from all tasks.
        
        Returns:
            List of output column names
        """
        return [task["output_column"] for task in self.tasks]
    
    def process_row(self, row: pd.Series, row_index: int) -> Dict[str, Any]:
        """
        Process a single row, executing all tasks.
        
        Args:
            row: DataFrame row to process
            row_index: Index of the row (for logging)
        
        Returns:
            Dictionary mapping output_column -> generated_value
        """
        results = {}
        
        for task in self.tasks:
            output_col = task["output_column"]
            input_col = task["input_column"]
            prompt_template = task["prompt_template"]
            
            # Skip if output already exists and is not empty
            if pd.notna(row.get(output_col)) and str(row.get(output_col)).strip():
                logger.debug(f"Row {row_index}: {output_col} already has value, skipping")
                continue
            
            # Get input text
            input_text = row.get(input_col)
            if pd.isna(input_text) or not str(input_text).strip():
                logger.debug(f"Row {row_index}: {input_col} is empty, skipping {output_col}")
                continue
            
            # Format prompt - support both {input} and {} formats
            try:
                # Try new format first: {input}
                prompt = prompt_template.format(input=input_text)
            except (KeyError, ValueError):
                try:
                    # Fallback to old format: {}
                    prompt = prompt_template.format(input_text)
                except (KeyError, ValueError):
                    # If both fail, just use the template as-is (might have custom placeholders)
                    logger.warning(f"Row {row_index}: Could not format prompt template, using as-is")
                    prompt = prompt_template.replace("{input}", str(input_text)).replace("{}", str(input_text))
            
            # Generate content
            logger.debug(f"Row {row_index}: Generating {output_col}")
            generated = self.api_client.generate(prompt)
            
            if generated:
                results[output_col] = generated.strip()
                logger.debug(f"Row {row_index}: Successfully generated {output_col}")
            else:
                logger.warning(f"Row {row_index}: Failed to generate {output_col}")
        
        return results
    
    def process_batch(self, df: pd.DataFrame, batch_start: int, batch_end: int) -> pd.DataFrame:
        """
        Process a batch of rows.
        
        Args:
            df: DataFrame to process
            batch_start: Start index (inclusive)
            batch_end: End index (exclusive)
        
        Returns:
            Updated DataFrame
        """
        df = df.copy()
        
        for idx in range(batch_start, batch_end):
            if idx >= len(df):
                break
            
            row = df.iloc[idx]
            results = self.process_row(row, idx)
            
            # Update DataFrame with results
            for output_col, value in results.items():
                df.at[idx, output_col] = value
        
        return df

