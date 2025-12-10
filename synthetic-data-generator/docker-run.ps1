# PowerShell script for running Synthetic Data Generator with Docker

param(
    [string]$Config = "configs/default.yaml",
    [Parameter(Mandatory=$false)]
    [string]$File = "",
    [int]$StartIndex = -1,
    [int]$EndIndex = -1,
    [int]$SaveIntervalRows = -1,
    [switch]$Status,
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: .\docker-run.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Config FILE              Configuration file (default: configs/default.yaml)"
    Write-Host "  -File FILE               CSV file to process"
    Write-Host "  -StartIndex N            Start processing from row N"
    Write-Host "  -EndIndex N              End processing at row N"
    Write-Host "  -SaveIntervalRows N      Save checkpoint every N rows"
    Write-Host "  -Status                  Show status snapshot"
    Write-Host "  -Help                    Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\docker-run.ps1 -File data/input/file.csv"
    Write-Host "  .\docker-run.ps1 -File data/input/file.csv -StartIndex 0 -EndIndex 1000"
    Write-Host "  .\docker-run.ps1 -File data/input/file.csv -Status"
    exit 0
}

# Check if file is provided
if ([string]::IsNullOrEmpty($File) -and -not $Status) {
    Write-Host "Error: CSV file is required" -ForegroundColor Red
    Write-Host "Use -Help for usage information"
    exit 1
}

# Build command
if ($Status) {
    $command = "python -m src.cli.status $File"
} else {
    $command = "python -m src.cli.main --config $Config"
    
    if ($StartIndex -ge 0) {
        $command += " --start-index $StartIndex"
    }
    
    if ($EndIndex -ge 0) {
        $command += " --end-index $EndIndex"
    }
    
    if ($SaveIntervalRows -gt 0) {
        $command += " --save-interval-rows $SaveIntervalRows"
    }
    
    $command += " $File"
}

# Print command
Write-Host "Running: $command" -ForegroundColor Blue
Write-Host ""

# Run Docker command
docker-compose run --rm synthetic-data-generator $command

