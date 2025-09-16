# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a QQQ (Nasdaq-100) index change alert system that runs as a GitHub Action. It monitors daily changes in the Nasdaq-100 index composition using official Invesco QQQ ETF holdings data, and sends beautifully formatted email notifications when changes occur, with special focus on the top 20 positions.

## Key Commands

### Running the QQQ Monitor
```bash
python scripts/check_changes.py
```
This will:
- Download current QQQ holdings from Invesco
- Parse and validate the data with multiple strategies
- Compare against historical data to detect changes
- Send HTML email alerts for significant changes
- Archive data with validation reports

### Local Development Tools
```bash
# Analyze ETF Excel files
python tools/etf_excel_converter.py --mode inspect --input data/raw/spy_holdings_raw.xlsx
python tools/etf_excel_converter.py --mode convert --input data/raw/spy_holdings_raw.xlsx
python tools/etf_excel_converter.py --mode batch --input data/raw

# Download and analyze ETF holdings
python notebooks/etf_holdings_analyzer.py
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Core Components

1. **Enhanced Main Script** (`scripts/check_changes.py`):
   - Downloads QQQ holdings from Invesco with multiple parsing strategies
   - Advanced data validation with comprehensive quality checks
   - Intelligent change detection with improved accuracy
   - Modern HTML email alerts with visual categorization
   - Real-time market cap integration (Alpha Vantage)
   - Detailed logging and progress reporting

2. **Enhanced Data Processing Pipeline**:
   - QQQ data from Invesco: `https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ`
   - Multi-strategy parsing: Excel, CSV, multiple encodings
   - Smart column detection and symbol standardization
   - Weight normalization with intelligent format detection
   - Comprehensive validation: duplicates, ranges, completeness
   - Alpha Vantage integration for top 15 holdings market cap data

3. **Caching System**:
   - API responses cached in `data/cache/` for 24 hours
   - Raw ETF files preserved in `data/raw/` for troubleshooting

4. **Optimized GitHub Actions Workflow** (`.github/workflows/daily_check.yml`):
   - Renamed to "Daily QQQ Change Check" for clarity
   - Runs weekdays at 6:00 PM UTC (after US market close)
   - Enhanced dependency verification with package checking
   - Improved logging with timestamps and progress indicators
   - Archives QQQ analysis data and validation reports
   - Smart commit messages with timestamps
   - 14-day retention for better troubleshooting

### Data Structure

- `data/qqq_current.json`: Current QQQ composition with weights, rankings, and market cap
- `data/shares_outstanding.json`: Market cap reference data
- `data/cache/`: 24-hour cached API responses with smart rate limiting
- `data/raw/`: Raw Invesco files, parsed data with validation reports, and analysis artifacts

### Environment Variables Required

- `EMAIL_SENDER`: Gmail address for sending modern HTML alerts
- `EMAIL_APP_PASSWORD`: Gmail app password (not regular password)
- `EMAIL_RECIPIENT`: Email address to receive beautifully formatted alerts
- `ALPHA_VANTAGE_API_KEY` (recommended): For real-time market cap data on top 15 QQQ holdings

## Key Configuration Constants

In `scripts/check_changes.py`:
- `TOP_POSITIONS_TO_TRACK = 20`: Top positions for priority alerting
- `MAX_WEIGHT_THRESHOLD = 25.0`: Maximum realistic individual weight (QQQ validation)
- `MAX_REASONABLE_CHANGES = 50`: Bulk change detection threshold
- Weight change threshold: 0.1% minimum for alerts
- Market cap change threshold: 5% minimum for reporting
- API rate limiting: 12 seconds between Alpha Vantage requests

## Development Notes

### Enhanced QQQ Data Processing
The system provides robust QQQ data handling through:
- Multi-format parsing: Excel, CSV, text with multiple encodings
- Smart column detection that adapts to Invesco format changes
- Advanced weight normalization with intelligent format detection
- Comprehensive symbol standardization and cleaning
- Extensive validation with detailed quality reports
- Progressive fallback strategies with detailed logging

### Enhanced Error Handling
- Smart API rate limiting with 24-hour caching
- Modern HTML email alerts with priority-based sending
- Comprehensive data archiving with validation reports
- Detailed progress logging with visual indicators
- Graceful degradation with informative error messages
- Automatic fallback to hardcoded QQQ data when needed

### Testing & Validation
- Manual trigger via "Daily QQQ Change Check" workflow in GitHub Actions
- Enhanced dependency verification with detailed package checking
- Comprehensive data validation reports with quality metrics
- Raw data preservation in Actions artifacts for 14 days
- Detailed logging helps troubleshoot parsing and validation issues
- Built-in data quality checks ensure reliable change detection