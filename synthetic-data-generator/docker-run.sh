#!/bin/bash
# Quick Docker run script for Synthetic Data Generator

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONFIG="configs/default.yaml"
FILE=""
START_INDEX=""
END_INDEX=""
SAVE_INTERVAL_ROWS=""
SHOW_STATUS=false
COMMAND=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --file|-f)
            FILE="$2"
            shift 2
            ;;
        --start-index)
            START_INDEX="$2"
            shift 2
            ;;
        --end-index)
            END_INDEX="$2"
            shift 2
            ;;
        --save-interval-rows)
            SAVE_INTERVAL_ROWS="$2"
            shift 2
            ;;
        --status)
            SHOW_STATUS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config, -c FILE          Configuration file (default: configs/default.yaml)"
            echo "  --file, -f FILE            CSV file to process"
            echo "  --start-index N            Start processing from row N"
            echo "  --end-index N              End processing at row N"
            echo "  --save-interval-rows N     Save checkpoint every N rows"
            echo "  --status                   Show status snapshot"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --file data/input/file.csv"
            echo "  $0 --file data/input/file.csv --start-index 0 --end-index 1000"
            echo "  $0 --file data/input/file.csv --status"
            exit 0
            ;;
        *)
            if [ -z "$FILE" ]; then
                FILE="$1"
            else
                echo "Unknown option: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if file is provided
if [ -z "$FILE" ] && [ "$SHOW_STATUS" = false ]; then
    echo "Error: CSV file is required"
    echo "Use --help for usage information"
    exit 1
fi

# Build command
if [ "$SHOW_STATUS" = true ]; then
    COMMAND="python -m src.cli.status $FILE"
else
    COMMAND="python -m src.cli.main --config $CONFIG"
    
    if [ -n "$START_INDEX" ]; then
        COMMAND="$COMMAND --start-index $START_INDEX"
    fi
    
    if [ -n "$END_INDEX" ]; then
        COMMAND="$COMMAND --end-index $END_INDEX"
    fi
    
    if [ -n "$SAVE_INTERVAL_ROWS" ]; then
        COMMAND="$COMMAND --save-interval-rows $SAVE_INTERVAL_ROWS"
    fi
    
    COMMAND="$COMMAND $FILE"
fi

# Print command
echo -e "${BLUE}Running:${NC} $COMMAND"
echo ""

# Run Docker command
docker-compose run --rm synthetic-data-generator $COMMAND

