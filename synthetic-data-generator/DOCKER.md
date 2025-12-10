# Docker Guide

Run the Synthetic Data Generator anywhere with Docker!

## 🐳 Quick Start

### Prerequisites

- Docker installed ([Get Docker](https://www.docker.com/get-started))
- Docker Compose (usually included with Docker Desktop)

### Quick Run Scripts

We provide convenient scripts for easy Docker usage:

**Linux/Mac:**
```bash
# Make script executable
chmod +x docker-run.sh

# Run processing
./docker-run.sh --file data/input/file.csv

# With options
./docker-run.sh --file data/input/file.csv --start-index 0 --end-index 1000 --save-interval-rows 100

# Check status
./docker-run.sh --file data/input/file.csv --status
```

**Windows (PowerShell):**
```powershell
# Run processing
.\docker-run.ps1 -File data/input/file.csv

# With options
.\docker-run.ps1 -File data/input/file.csv -StartIndex 0 -EndIndex 1000 -SaveIntervalRows 100

# Check status
.\docker-run.ps1 -File data/input/file.csv -Status
```

### 1. Build the Image

```bash
# Build the Docker image
docker build -t synthetic-data-generator:latest .

# Or using docker-compose
docker-compose build
```

### 2. Run with Docker

```bash
# Run a command
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/.env:/app/.env:ro \
  synthetic-data-generator:latest \
  python -m src.cli.main --config configs/default.yaml data/input/file.csv
```

### 3. Run with Docker Compose

```bash
# Build and run
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.main --config configs/default.yaml data/input/file.csv

# Or override command in docker-compose.yml
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.main --config configs/default.yaml --start-index 0 --end-index 1000 data/input/file.csv
```

## 📋 Common Usage Examples

### Process a File

```bash
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.main --config configs/default.yaml data/input/file.csv
```

### Process with Index Range

```bash
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.main \
    --config configs/default.yaml \
    --start-index 0 \
    --end-index 1000 \
    data/input/file.csv
```

### Process with Checkpoints

```bash
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.main \
    --config configs/default.yaml \
    --save-interval-rows 100 \
    data/input/file.csv
```

### Check Status

```bash
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.status data/input/file.csv
```

### Interactive Shell

```bash
# Get an interactive shell in the container
docker-compose run --rm synthetic-data-generator /bin/bash

# Then run commands normally
python -m src.cli.main --config configs/default.yaml data/input/file.csv
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
API_KEYS=your_key_1,your_key_2,your_key_3
```

The `.env` file is automatically mounted as read-only in docker-compose.

### Volume Mounts

The docker-compose.yml mounts:
- `./data` → `/app/data` (input, output, progress, checkpoints, snapshots)
- `./configs` → `/app/configs` (configuration files)
- `./logs` → `/app/logs` (log files)
- `./.env` → `/app/.env` (environment variables, read-only)

### Custom Configuration

Edit `docker-compose.yml` or create `docker-compose.override.yml`:

```yaml
version: '3.8'

services:
  synthetic-data-generator:
    command: ["python", "-m", "src.cli.main", "--config", "configs/my_config.yaml", "data/input/file.csv"]
    environment:
      - API_KEYS=custom_key_1,custom_key_2
```

## 🚀 Production Usage

### Build for Production

```bash
# Build with specific tag
docker build -t synthetic-data-generator:v1.0.0 .

# Tag for registry
docker tag synthetic-data-generator:v1.0.0 your-registry/synthetic-data-generator:v1.0.0

# Push to registry
docker push your-registry/synthetic-data-generator:v1.0.0
```

### Run in Production

```bash
# Pull and run
docker run --rm \
  -v /path/to/data:/app/data \
  -v /path/to/configs:/app/configs \
  -v /path/to/.env:/app/.env:ro \
  your-registry/synthetic-data-generator:v1.0.0 \
  python -m src.cli.main --config configs/default.yaml data/input/file.csv
```

## 🔄 Parallel Processing with Docker

Run multiple containers in parallel:

```bash
# Terminal 1: Process rows 0-1000
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.main \
    --config configs/default.yaml \
    --start-index 0 \
    --end-index 1000 \
    data/input/file.csv

# Terminal 2: Process rows 1000-2000
docker-compose run --rm synthetic-data-generator \
  python -m src.cli.main \
    --config configs/default.yaml \
    --start-index 1000 \
    --end-index 2000 \
    data/input/file.csv
```

All containers share the same mounted volumes, so they update the same working files safely.

## 🐛 Troubleshooting

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Fix permissions (Linux/Mac)
sudo chown -R $USER:$USER data/ logs/

# Or run container with your user ID
docker run --rm \
  -u $(id -u):$(id -g) \
  -v $(pwd)/data:/app/data \
  synthetic-data-generator:latest \
  python -m src.cli.main --config configs/default.yaml data/input/file.csv
```

### Environment Variables Not Loading

Make sure `.env` file exists and is mounted:

```bash
# Check if .env is mounted
docker-compose run --rm synthetic-data-generator cat /app/.env

# Or pass environment variables directly
docker run --rm \
  -e API_KEYS=key1,key2 \
  -v $(pwd)/data:/app/data \
  synthetic-data-generator:latest \
  python -m src.cli.main --config configs/default.yaml data/input/file.csv
```

### Container Keeps Exiting

Use interactive mode:

```bash
docker-compose run --rm synthetic-data-generator /bin/bash
```

### View Logs

```bash
# View logs from docker-compose
docker-compose logs -f

# View logs from docker run
docker logs <container_id>
```

## 📦 Image Size Optimization

The current image uses `python:3.11-slim` for a smaller footprint. For even smaller images, consider:

```dockerfile
# Multi-stage build example
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
```

## 🔐 Security Best Practices

1. **Never commit `.env` files** - Use secrets management in production
2. **Use read-only mounts** for `.env` files (`:ro` flag)
3. **Limit container resources** in production
4. **Use specific image tags** instead of `latest` in production
5. **Scan images** for vulnerabilities: `docker scan synthetic-data-generator:latest`

## 📝 Notes

- All data persists in mounted volumes on the host
- Logs are written to mounted `logs/` directory
- Checkpoints and snapshots are saved in mounted `data/` directories
- The container is stateless - all state is in mounted volumes

