# Synthetic Data Generator

A modular, production-ready system for generating synthetic data using AI APIs. Built with clean architecture, comprehensive error handling, and user-friendly configuration.

## ✨ Features

- **🔧 Modular Architecture**: Clean separation of concerns, easy to extend
- **🔄 Multi-API Key Support**: Automatic rotation and usage tracking
- **💾 Progress Tracking**: Resume from interruptions, never lose progress
- **⚙️ YAML Configuration**: Simple, readable configuration files
- **📊 Batch Processing**: Efficient processing of large datasets
- **🛡️ Robust Error Handling**: Comprehensive retry logic and error recovery
- **📝 Comprehensive Logging**: Detailed logs for debugging and monitoring
- **🎯 Task-Based System**: Define multiple tasks per row with custom prompts

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**Quick Start:**
```bash
# Build the image
docker build -t synthetic-data-generator:latest .

# Run with helper script (Linux/Mac)
./docker-run.sh --file data/input/file.csv

# Or with docker-compose
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.main --config configs/default.yaml data/input/file.csv
```

**Windows (PowerShell):**
```powershell
.\docker-run.ps1 -File data/input/file.csv
```

See [DOCKER.md](DOCKER.md) for detailed Docker instructions.

### Option 2: Local Installation

```bash
# Clone or navigate to the project directory
cd synthetic-data-generator

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# API_KEYS=key1,key2,key3
```

### 3. Create Configuration File

Create a YAML config file (or use the examples in `configs/examples/`):

```yaml
api:
  provider: gemini
  model: gemini-2.5-flash
  keys: ${API_KEYS}  # From .env file

processing:
  batch_size: 100
  save_interval: 1
  retry_delay: 0.1
  save_interval_rows: 100  # Save checkpoint every 100 rows

tasks:
  - name: translate_question
    input_column: Question
    output_column: Question_Bangla
    prompt_template: |
      Translate to Bangla:
      {input}
```

### 4. Run

```bash
# Process a single file
python -m src.cli.main --config configs/default.yaml data/input/your_file.csv

# Process a specific index range (for parallel processing)
python -m src.cli.main --config configs/default.yaml --start-index 0 --end-index 1000 data/input/file.csv

# Process with checkpoints every 100 rows
python -m src.cli.main --config configs/default.yaml --save-interval-rows 100 data/input/file.csv

# Process multiple files
python -m src.cli.main --config configs/default.yaml data/input/*.csv

# Override batch size
python -m src.cli.main --config configs/default.yaml --batch-size 50 data/input/file.csv
```

## 📁 Project Structure

```
synthetic-data-generator/
├── src/
│   ├── core/              # Core processing logic
│   │   ├── api_client.py      # API key management & rotation
│   │   ├── progress_tracker.py # Progress tracking & resumability
│   │   ├── batch_processor.py  # Batch processing engine
│   │   └── task_executor.py    # Task execution engine
│   ├── providers/         # AI API providers
│   │   ├── base.py            # Base provider interface
│   │   └── gemini.py         # Google Gemini implementation
│   ├── config/            # Configuration management
│   │   ├── loader.py          # Config file loader
│   │   └── validator.py       # Config validation
│   ├── utils/             # Utilities
│   │   ├── logger.py          # Logging setup
│   │   └── file_handler.py    # File operations
│   └── cli/               # Command-line interface
│       └── main.py            # CLI entry point
├── configs/               # Configuration files
│   ├── default.yaml
│   └── examples/
├── data/
│   ├── input/             # Input CSV files
│   ├── output/            # Generated output files
│   └── progress/          # Progress tracking files
└── logs/                  # Log files
```

## ⚙️ Configuration

### API Configuration

```yaml
api:
  provider: gemini          # Provider name (gemini, openai, etc.)
  model: gemini-2.5-flash   # Model name
  keys: ${API_KEYS}         # Environment variable or comma-separated keys
```

### Processing Configuration

```yaml
processing:
  batch_size: 100           # Rows per batch
  save_interval: 1          # Save after every N batches
  retry_delay: 0.1          # Delay between API requests (seconds)
  max_retries: 3            # Maximum retries per request
  save_interval_rows: 100   # Save checkpoint after every N rows (null to disable)
  checkpoint_dir: data/checkpoints  # Directory for checkpoint saves
```

### Task Configuration

```yaml
tasks:
  - name: task_name          # Unique task name
    input_column: Column1    # Source column name
    output_column: Column2   # Destination column name
    prompt_template: |       # Prompt template
      Your prompt here.
      Use {input} for the input value.
```

## 🎯 Usage Examples

### Medical Data Translation

```bash
python -m src.cli.main \
  --config configs/examples/medical_translation.yaml \
  data/input/medical_data.csv
```

### Song Generation

```bash
python -m src.cli.main \
  --config configs/examples/song_generation.yaml \
  data/input/songs.csv
```

### Parallel Processing with Index Ranges

Process different parts of a large file in parallel on different machines/terminals:

```bash
# Terminal 1: Process rows 0-1000
python -m src.cli.main \
  --config configs/default.yaml \
  --start-index 0 --end-index 1000 \
  data/input/large_file.csv

# Terminal 2: Process rows 1000-2000
python -m src.cli.main \
  --config configs/default.yaml \
  --start-index 1000 --end-index 2000 \
  data/input/large_file.csv

# Terminal 3: Process rows 2000-3000
python -m src.cli.main \
  --config configs/default.yaml \
  --start-index 2000 --end-index 3000 \
  data/input/large_file.csv
```

All processes will update the same working file safely!

### Processing with Frequent Checkpoints

```bash
# Save checkpoint every 100 rows for safety
python -m src.cli.main \
  --config configs/default.yaml \
  --save-interval-rows 100 \
  data/input/file.csv
```

Checkpoints are saved in `data/checkpoints/` with timestamps and row counts.

### Custom Configuration

1. Copy an example config:
   ```bash
   cp configs/examples/medical_translation.yaml configs/my_config.yaml
   ```

2. Edit `my_config.yaml` with your tasks and prompts

3. Run:
   ```bash
   python -m src.cli.main --config configs/my_config.yaml data/input/file.csv
   ```

## 🔄 Progress Tracking & Resumability

The system automatically tracks progress for each CSV file:

- **Progress files** are saved in `data/progress/`
- **Working files** (`working_*.csv`) are created automatically
- **Checkpoints** are saved periodically in `data/checkpoints/` (if enabled)
- If interrupted, simply re-run the same command to resume
- Progress is saved after each row, so minimal data loss
- **Atomic writes** ensure data integrity (writes to temp file, then renames)
- **Range processing** supports parallel execution on different index ranges

## 📊 API Key Management

- **Multiple keys**: Add comma-separated keys in `.env` or config
- **Automatic rotation**: Keys rotate on failure or rate limits
- **Usage tracking**: See which keys are used and how many requests
- **Health monitoring**: Failed keys are automatically skipped

## 🛠️ Advanced Usage

### Custom Output Directory

```bash
python -m src.cli.main \
  --config configs/default.yaml \
  --output-dir custom/output \
  data/input/file.csv
```

### Debug Mode

```bash
python -m src.cli.main \
  --config configs/default.yaml \
  --log-level DEBUG \
  data/input/file.csv
```

### Override Batch Size

```bash
python -m src.cli.main \
  --config configs/default.yaml \
  --batch-size 50 \
  data/input/file.csv
```

### Process Index Range

```bash
# Process only rows 500-1000
python -m src.cli.main \
  --config configs/default.yaml \
  --start-index 500 \
  --end-index 1000 \
  data/input/file.csv
```

### Enable Checkpoints

```bash
# Save checkpoint every 100 rows
python -m src.cli.main \
  --config configs/default.yaml \
  --save-interval-rows 100 \
  data/input/file.csv
```

## 📊 Status & Snapshots

### View Processing Status

Check which row ranges have been completed:

```bash
# Show status for a file
python -m src.cli.status data/input/file.csv

# Show status for multiple files
python -m src.cli.status data/input/*.csv
```

### Status Output

The status command shows:
- ✅ **Completed ranges**: Which row ranges are done
- ⏳ **Pending ranges**: Which rows still need processing
- 📈 **Visual progress bar**: Graphical representation
- 📊 **Statistics**: Completion percentage, row counts

### Show Status Before Processing

```bash
python -m src.cli.main \
  --config configs/default.yaml \
  --show-status \
  data/input/file.csv
```

### Snapshot Files

Snapshots are automatically saved in `data/snapshots/`:
- Updated when ranges are completed
- JSON format for easy inspection
- Used to track progress across parallel processes

## 🔍 Logging

Logs are written to:
- **Console**: Real-time progress and status
- **File**: Detailed logs in `logs/synthetic_data_YYYYMMDD_HHMMSS.log`

Set log level with `--log-level`:
- `DEBUG`: Detailed debugging information
- `INFO`: General information (default)
- `WARNING`: Warnings only
- `ERROR`: Errors only

## 🐳 Docker Usage

The project includes full Docker support for easy deployment anywhere.

### Quick Docker Commands

```bash
# Build image
docker build -t synthetic-data-generator:latest .

# Process a file
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.main --config configs/default.yaml data/input/file.csv

# Check status
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.status data/input/file.csv
```

See [DOCKER.md](DOCKER.md) for comprehensive Docker documentation.

## 🐛 Troubleshooting

### "No API keys found"
- Check your `.env` file exists and contains `API_KEYS=...`
- Ensure keys are comma-separated without spaces

### "Configuration validation failed"
- Check YAML syntax
- Ensure all required fields are present (see config validator)

### "All API keys exhausted"
- Check API key validity
- Verify API quotas/limits
- Add more keys to rotation

### Processing interrupted
- Simply re-run the same command
- System will automatically resume from last checkpoint

## 🚧 Future Enhancements

- [ ] Support for OpenAI, Anthropic, and other providers
- [ ] Parallel processing for faster execution
- [ ] Web UI for easier interaction
- [ ] Data validation and quality checks
- [ ] Export to multiple formats (JSON, Parquet, etc.)
- [ ] Cost estimation and tracking

## 📝 License

This project is provided as-is for synthetic data generation purposes.

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows existing patterns
- Tests are added for new features
- Documentation is updated

## 📧 Support

For issues or questions, please check:
1. Configuration file syntax
2. API key validity
3. Log files for detailed error messages

