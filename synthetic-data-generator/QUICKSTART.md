# Quick Start Guide

Get up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys (comma-separated)
# API_KEYS=your_key_1,your_key_2,your_key_3
```

## Step 3: Prepare Your Data

Place your CSV file(s) in `data/input/` directory.

Your CSV should have columns that match the `input_column` names in your config.

## Step 4: Choose or Create a Config

### Option A: Use Example Config

```bash
# For medical translation
python -m src.cli.main --config configs/examples/medical_translation.yaml data/input/your_file.csv

# For song generation
python -m src.cli.main --config configs/examples/song_generation.yaml data/input/your_file.csv
```

### Option B: Use Default Config

```bash
python -m src.cli.main --config configs/default.yaml data/input/your_file.csv
```

### Option C: Create Custom Config

1. Copy an example:
   ```bash
   cp configs/examples/medical_translation.yaml configs/my_config.yaml
   ```

2. Edit `configs/my_config.yaml`:
   - Update `input_column` to match your CSV columns
   - Update `output_column` to desired output column names
   - Customize `prompt_template` for your use case

3. Run:
   ```bash
   python -m src.cli.main --config configs/my_config.yaml data/input/your_file.csv
   ```

## Step 5: Monitor Progress

- **Console**: Real-time progress bars and status updates
- **Logs**: Detailed logs in `logs/` directory
- **Progress Files**: Check `data/progress/` for resume information

## Step 6: Get Results

Output files are saved in `data/output/` with the prefix `final_`.

## Common Issues

### "No API keys found"
- Make sure `.env` file exists in the project root
- Check that `API_KEYS=...` is set correctly
- Keys should be comma-separated: `API_KEYS=key1,key2,key3`

### "Configuration validation failed"
- Check YAML syntax (indentation matters!)
- Ensure all required fields are present
- Use a YAML validator if needed

### Processing interrupted?
- Just re-run the same command
- The system will automatically resume from where it stopped

## Check Processing Status

See which row ranges have been completed:

```bash
# Check status
python -m src.cli.status data/input/your_file.csv
```

This shows:
- ✅ Completed ranges
- ⏳ Pending ranges  
- 📊 Completion percentage
- 📈 Visual progress bar

## Next Steps

- Read the full [README.md](README.md) for advanced features
- Check [ARCHITECTURE_PLAN.md](../ARCHITECTURE_PLAN.md) for system design
- Check [FEATURES.md](FEATURES.md) for detailed feature documentation
- Customize configs for your specific use case

