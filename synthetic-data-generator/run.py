#!/usr/bin/env python3
"""
Simple wrapper script for running the synthetic data generator.
Usage: python run.py --config configs/default.yaml data/input/file.csv
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli.main import main

if __name__ == "__main__":
    main()

