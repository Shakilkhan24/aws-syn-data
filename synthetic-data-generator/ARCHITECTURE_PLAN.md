# Synthetic Data Generation System - Architecture Plan

## 🎯 Goals
- **Modular**: Clean separation of concerns, easy to extend
- **User-Friendly**: Simple configuration, clear CLI, helpful error messages
- **Robust**: Comprehensive error handling, progress tracking, resumability
- **Scalable**: Support multiple API providers, batch processing, parallel execution
- **Professional**: Proper logging, documentation, type hints

## 📁 Project Structure
```
synthetic-data-generator/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── api_client.py          # API client manager with rotation
│   │   ├── progress_tracker.py    # Progress tracking and resumability
│   │   ├── batch_processor.py     # Batch processing logic
│   │   └── task_executor.py       # Task execution engine
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                # Base provider interface
│   │   ├── gemini.py              # Google Gemini provider
│   │   └── openai.py              # OpenAI provider (future)
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py              # Config file loader
│   │   └── validator.py           # Config validation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py              # Logging setup
│   │   └── file_handler.py        # File operations
│   └── cli/
│       ├── __init__.py
│       └── main.py                # CLI entry point
├── configs/
│   ├── default.yaml               # Default configuration
│   └── examples/
│       ├── medical_translation.yaml
│       └── song_generation.yaml
├── data/
│   ├── input/                     # Input CSV files
│   ├── output/                    # Generated output files
│   └── progress/                  # Progress tracking files
├── tests/
│   └── test_core.py
├── .env.example                   # Environment variables template
├── requirements.txt
├── setup.py
└── README.md
```

## 🔧 Core Components

### 1. Configuration System
- **YAML-based config files** for easy customization
- **Environment variable support** for sensitive data (API keys)
- **Config validation** with helpful error messages
- **Multiple config profiles** for different use cases

### 2. API Client Manager
- **Multi-provider support** (Gemini, OpenAI, etc.)
- **Automatic key rotation** with usage tracking
- **Rate limiting** and retry logic
- **Health monitoring** for API keys

### 3. Progress Tracker
- **JSON-based progress files** per CSV file
- **Resume capability** from last checkpoint
- **Progress visualization** with tqdm
- **Automatic cleanup** on completion

### 4. Batch Processor
- **Configurable batch sizes**
- **Parallel processing** support (future)
- **Memory-efficient** processing
- **Automatic saving** after each batch

### 5. Task Executor
- **Plugin-based task system**
- **Multiple tasks per row** support
- **Custom prompt templates**
- **Output validation**

### 6. CLI Interface
- **User-friendly commands**
- **Interactive mode** for first-time users
- **Progress indicators**
- **Error reporting**

## 🚀 Features

### Phase 1 (Core)
- ✅ Multi-API key management
- ✅ Progress tracking & resumability
- ✅ Batch processing
- ✅ YAML configuration
- ✅ CLI interface
- ✅ Comprehensive logging

### Phase 2 (Enhanced)
- 🔄 Multiple API provider support
- 🔄 Parallel processing
- 🔄 Web UI (optional)
- 🔄 Data validation
- 🔄 Export to multiple formats

## 📝 Configuration Example

```yaml
# configs/medical_translation.yaml
api:
  provider: gemini
  model: gemini-2.5-flash
  keys: ${API_KEYS}  # From .env
  rate_limit: 10  # requests per second

processing:
  batch_size: 100
  max_retries: 3
  retry_delay: 2  # seconds
  save_interval: 1  # batches

tasks:
  - name: translate_question
    input_column: Question
    output_column: Question_Bangla
    prompt_template: |
      আপনি একজন দক্ষ চিকিৎসা অনুবাদক...
      ইংরেজি প্রশ্ন: {input}

  - name: translate_reasoning
    input_column: Complex_CoT
    output_column: Complex_CoT_Bangla
    prompt_template: |
      ...
```

## 🎨 User Experience

1. **Setup**: Copy `.env.example` to `.env`, add API keys
2. **Configure**: Create/edit YAML config file
3. **Run**: `synthetic-data generate --config configs/medical_translation.yaml data/input/file.csv`
4. **Monitor**: Real-time progress, automatic saving
5. **Resume**: Automatically resumes if interrupted

