# New Features: Index Range Processing, Checkpoints & Snapshots

## 📸 Snapshot System

Track which row ranges have been completed with an easy-to-read snapshot system.

### Features

- **Automatic Tracking**: Ranges are automatically marked as completed
- **Visual Status**: See completed vs pending ranges at a glance
- **Progress Visualization**: Visual progress bar showing completion
- **Statistics**: Completion percentage, row counts, range counts
- **Standalone Command**: Check status anytime without processing

### Usage

```bash
# Check status of a file
python -m src.cli.status data/input/file.csv

# Check status of multiple files
python -m src.cli.status data/input/*.csv

# Show status before processing
python -m src.cli.main --config configs/default.yaml --show-status data/input/file.csv
```

### Status Output Example

```
======================================================================
📊 SNAPSHOT: medical_data
======================================================================
Total Rows: 10,000
Completed: 7,500 (75.00%)
Pending: 2,500 (25.00%)

✅ Completed Ranges (3):
   [0 - 2,499] (2,500 rows)
   [2,500 - 4,999] (2,500 rows)
   [5,000 - 7,499] (2,500 rows)

⏳ Pending Ranges (1):
   [7,500 - 9,999] (2,500 rows)

📈 Visual Progress:
   [████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░]
   0                   5,000                          10,000

Last Updated: 2024-12-10T17:30:45
======================================================================
```

### How It Works

- **Snapshot Files**: Stored in `data/snapshots/` as JSON
- **Automatic Updates**: Updated when ranges complete
- **Range Merging**: Overlapping ranges are automatically merged
- **Persistent**: Survives restarts and parallel processing

### Benefits

1. **Easy Monitoring**: Quickly see what's done and what's left
2. **Parallel Coordination**: Know which ranges to assign to new processes
3. **Progress Tracking**: Visual representation of completion
4. **Resume Planning**: Identify exactly which ranges need processing

## 🎯 Index Range Processing

## 🎯 Index Range Processing

Process specific index ranges of a CSV file, enabling parallel processing on different machines or terminals.

### Use Cases

1. **Parallel Processing**: Split a large file across multiple machines/processes
2. **Selective Processing**: Only process certain rows
3. **Resume Specific Range**: Resume processing a specific section

### Usage

```bash
# Process rows 0-1000
python -m src.cli.main \
  --config configs/default.yaml \
  --start-index 0 \
  --end-index 1000 \
  data/input/file.csv

# Process rows 1000-2000 (in parallel)
python -m src.cli.main \
  --config configs/default.yaml \
  --start-index 1000 \
  --end-index 2000 \
  data/input/file.csv
```

### How It Works

- **Index Range**: `--start-index` (inclusive) and `--end-index` (exclusive)
- **Shared Working File**: All processes update the same `working_*.csv` file
- **Range-Specific Output**: Each process creates its own output file with range suffix
- **Progress Tracking**: Separate progress files per range (prevents conflicts)

### Parallel Processing Example

```bash
# Machine 1: Process first 1000 rows
python -m src.cli.main --config configs/default.yaml --start-index 0 --end-index 1000 file.csv &

# Machine 2: Process next 1000 rows
python -m src.cli.main --config configs/default.yaml --start-index 1000 --end-index 2000 file.csv &

# Machine 3: Process next 1000 rows
python -m src.cli.main --config configs/default.yaml --start-index 2000 --end-index 3000 file.csv &
```

All processes safely update the same working file!

## 💾 Checkpoint System

Save progress at regular intervals for maximum safety.

### Features

- **Row-Based Checkpoints**: Save after every N rows (configurable)
- **Automatic Checkpoints**: On batch completion and errors
- **Timestamped Files**: Each checkpoint includes timestamp and row count
- **Safe Recovery**: Restore from any checkpoint if needed

### Configuration

```yaml
processing:
  save_interval_rows: 100  # Save checkpoint every 100 rows
  checkpoint_dir: data/checkpoints  # Checkpoint directory
```

### Usage

```bash
# Enable checkpoints via CLI
python -m src.cli.main \
  --config configs/default.yaml \
  --save-interval-rows 100 \
  data/input/file.csv
```

### Checkpoint Files

Checkpoints are saved as:
```
data/checkpoints/
  ├── file_checkpoint_r100_20241210_143022.csv
  ├── file_checkpoint_r200_20241210_143045.csv
  └── file_checkpoint_r300_20241210_143108.csv
```

Format: `{filename}_checkpoint_r{row_count}_{timestamp}.csv`

### Automatic Checkpoints

Checkpoints are automatically saved:
- ✅ Every N rows (if `save_interval_rows` is set)
- ✅ On batch completion
- ✅ On errors/interruptions (emergency checkpoint)
- ✅ On final completion

## 🛡️ Safety Features

### Atomic Writes

All file writes use atomic operations:
- Write to temporary file first
- Rename to final file (atomic on most systems)
- Prevents corruption on interruption

### Progress Tracking

- **Per-Range Progress**: Separate progress files for each range
- **Resume Support**: Automatically resume from last checkpoint
- **Row-Level Tracking**: Progress saved after every row

### Error Recovery

- **Emergency Checkpoints**: Saved on errors/interruptions
- **Working File Updates**: Always kept up-to-date
- **Progress Files**: Track exact resume point

## 📊 Example Workflow

### Scenario: Process 10,000 rows in parallel

```bash
# Terminal 1: Rows 0-2500
python -m src.cli.main \
  --config configs/default.yaml \
  --start-index 0 --end-index 2500 \
  --save-interval-rows 100 \
  data/input/large_file.csv

# Terminal 2: Rows 2500-5000
python -m src.cli.main \
  --config configs/default.yaml \
  --start-index 2500 --end-index 5000 \
  --save-interval-rows 100 \
  data/input/large_file.csv

# Terminal 3: Rows 5000-7500
python -m src.cli.main \
  --config configs/default.yaml \
  --start-index 5000 --end-index 7500 \
  --save-interval-rows 100 \
  data/input/large_file.csv

# Terminal 4: Rows 7500-10000
python -m src.cli.main \
  --config configs/default.yaml \
  --start-index 7500 --end-index 10000 \
  --save-interval-rows 100 \
  data/input/large_file.csv
```

### Results

- **Working File**: `working_large_file.csv` (updated by all processes)
- **Range Outputs**: 
  - `final_large_file_r0-2499.csv`
  - `final_large_file_r2500-4999.csv`
  - `final_large_file_r5000-7499.csv`
  - `final_large_file_r7500-9999.csv`
- **Checkpoints**: Multiple checkpoints in `data/checkpoints/`
- **Progress Files**: Separate progress files per range

## 🔧 Configuration Options

### CLI Arguments

- `--start-index`: Starting row index (0-based, inclusive)
- `--end-index`: Ending row index (0-based, exclusive)
- `--save-interval-rows`: Save checkpoint every N rows
- `--checkpoint-dir`: Custom checkpoint directory

### Config File Options

```yaml
processing:
  save_interval_rows: 100  # null to disable
  checkpoint_dir: data/checkpoints
```

## 💡 Best Practices

1. **Use Checkpoints**: Enable `save_interval_rows` for large files
2. **Parallel Processing**: Split large files into ranges for faster processing
3. **Monitor Checkpoints**: Check `data/checkpoints/` for progress
4. **Resume Safely**: Re-run same command to resume from interruption
5. **Range Overlap**: Avoid overlapping ranges (though system handles it safely)

## 🚨 Important Notes

- **Index Range**: `start_index` is inclusive, `end_index` is exclusive (like Python slicing)
- **Working File**: All processes share the same working file (atomic writes ensure safety)
- **Progress Files**: Each range has its own progress file (prevents conflicts)
- **Output Files**: Range-specific outputs are created separately
- **Checkpoints**: Include full DataFrame state at that point

