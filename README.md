# QQQ (Nasdaq-100) Change Alert System

This repository contains a GitHub Action that automatically tracks changes in the QQQ (Nasdaq-100) index using official Invesco ETF data. When changes occur, especially in the top 20 positions, the system sends beautifully formatted email notifications with detailed analysis and insights.

## Features

- Daily monitoring of QQQ (Nasdaq-100) index composition using **official Invesco ETF holdings data**
- Enhanced data parsing with multiple validation layers and intelligent error handling
- Detects additions, removals, rank changes, and significant weight changes with improved accuracy
- Special focus on the top 20 positions with priority-based alerting
- **Modern HTML email alerts** with responsive design and visual categorization
- **Enhanced market cap tracking** with real-time data integration (Alpha Vantage)
- **Smart caching system** to reduce API calls and improve reliability
- **Multi-strategy parsing** with intelligent fallback mechanisms
- **Comprehensive data validation** with detailed quality reports
- **Raw data preservation** and analysis artifacts for troubleshooting
- Completely free to run using GitHub Actions infrastructure

## Why This Approach?

This system uses official Invesco QQQ ETF holdings data, which provides several advantages:
- **Authoritative source**: Direct from the fund provider, not third-party APIs
- **No rate limits**: Core holdings data doesn't require API keys
- **Accurate weights**: Precise component weightings from the official source
- **Real rankings**: Official position rankings within the Nasdaq-100
- **Enhanced validation**: Multiple parsing strategies ensure data accuracy

## Setup Instructions

1. **Fork or clone this repository**

2. **Configure GitHub Secrets**

   Add the following secrets to your repository:
   
   - `EMAIL_SENDER`: Email address that will send notifications (Gmail recommended)
   - `EMAIL_APP_PASSWORD`: App password for your email (for Gmail: https://myaccount.google.com/apppasswords)
   - `EMAIL_RECIPIENT`: Email address where you want to receive the beautiful HTML alerts
   - `ALPHA_VANTAGE_API_KEY` (Optional but recommended): For real-time market cap data on top QQQ holdings

3. **Initial Run**

   After setting up secrets, manually trigger the workflow by going to the "Actions" tab in your repository, selecting "Daily QQQ Change Check" workflow, and clicking "Run workflow".

   This initial run will:
   - Download current QQQ holdings from Invesco
   - Validate and process the data
   - Create baseline files for future comparisons
   - Generate a validation report

4. **Automatic Execution**

   The workflow runs automatically at 6:00 PM UTC (after US market close) on weekdays, providing:
   - Fresh QQQ holdings data analysis
   - Change detection with enhanced accuracy
   - Beautiful HTML email alerts for any changes
   - Data archiving for troubleshooting

## How It Works

1. **Scheduled Execution**: GitHub Action runs weekdays at 6:00 PM UTC (after market close)
2. **Data Download**: Retrieves latest QQQ holdings from official Invesco source
3. **Enhanced Parsing**: Uses multiple parsing strategies with intelligent error handling
4. **Data Validation**: Comprehensive quality checks with detailed reporting
5. **Market Cap Enhancement**: Integrates real-time market data for top 15 holdings (optional)
6. **Change Detection**: Compares against previous data with improved accuracy algorithms
7. **Smart Alerting**: Sends modern HTML email alerts for significant changes
8. **Data Archiving**: Preserves raw data and analysis for troubleshooting and trends

## Repository Structure

```
index-change-alerts/
├── .github/workflows/        # GitHub Actions configuration
│   └── daily_check.yml      # QQQ monitoring workflow
├── data/                     # QQQ data storage and cache
│   ├── qqq_current.json     # Current QQQ composition
│   ├── shares_outstanding.json  # Market cap reference data
│   ├── cache/                # Cached API responses (24hr)
│   └── raw/                  # Raw Invesco files + parsed data
├── scripts/                  # Enhanced Python processing
│   └── check_changes.py      # Main script with advanced validation
├── tools/                    # Local development utilities
│   └── etf_excel_converter.py  # Excel analysis tool
├── notebooks/                # Development and testing
│   └── etf_holdings_analyzer.py  # QQQ data exploration
├── CLAUDE.md                 # AI assistant guidance
└── README.md                 # This documentation
```

## Data Source Integration

The system integrates with official Invesco QQQ ETF data:

### QQQ (Nasdaq-100 ETF) - Invesco
- **Data URL**: https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ
- **Update Frequency**: Daily after US market close (typically by 6 PM ET)
- **Format**: Excel file (.xlsx) with comprehensive holdings data
- **Content**: ~100 Nasdaq-100 components with weights, names, and rankings
- **Parsing**: Multiple strategies (standard Excel, CSV fallback, different encodings)
- **Validation**: Symbol cleaning, weight normalization, duplicate detection

### Optional Market Data Enhancement
- **Source**: Alpha Vantage API (free tier: 500 requests/day)
- **Purpose**: Real-time market cap data for top 15 QQQ holdings
- **Features**: Company names verification, market cap tracking
- **Rate Limiting**: 12-second delays between requests
- **Caching**: 24-hour cache to minimize API usage

## Advanced Data Processing

The system employs sophisticated data processing techniques:

### Enhanced Parsing Strategies
1. **Multi-Format Support**: Excel (.xlsx), CSV, and text formats
2. **Intelligent Column Detection**: Flexible column mapping for format changes
3. **Multiple Encoding Support**: UTF-8, Latin-1, CP1252 for international compatibility
4. **Smart Row Detection**: Automatic header detection and data validation

### Data Quality & Validation
1. **Symbol Standardization**: Cleans and validates ticker symbols
2. **Weight Normalization**: Intelligent percentage vs decimal detection
3. **Duplicate Detection**: Identifies and resolves duplicate entries
4. **Range Validation**: Ensures realistic weight and rank values
5. **Completeness Checks**: Validates expected QQQ characteristics (~100 components)

### Error Handling & Recovery
1. **Multiple Fallback Strategies**: Progressive parsing attempts
2. **Raw Data Preservation**: All downloads saved for debugging
3. **Detailed Logging**: Comprehensive processing reports
4. **Graceful Degradation**: Continues with partial data when possible
5. **Hardcoded Backup**: Updated fallback data for critical failures

## Local Development Tools

The repository includes tools for local development and troubleshooting:

### ETF Excel Converter

A utility to analyze and convert ETF provider Excel files:

```bash
# Inspect a QQQ Excel file
python tools/etf_excel_converter.py --mode inspect --input data/raw/qqq_holdings_raw.xlsx

# Convert to CSV for easier viewing
python tools/etf_excel_converter.py --mode convert --input data/raw/qqq_holdings_raw.xlsx

# Batch process all raw files
python tools/etf_excel_converter.py --mode batch --input data/raw
```

### QQQ Holdings Analyzer

A development script to download and explore QQQ holdings data:

```bash
# Run the analyzer to download and process QQQ holdings
python notebooks/etf_holdings_analyzer.py
```

## Customization

The system offers several customization options:

### Alert Customization
- **`TOP_POSITIONS_TO_TRACK`** (default: 20): Number of top positions to monitor with priority
- **Weight change threshold** (default: 0.1%): Minimum change to trigger weight alerts
- **Market cap change threshold** (default: 5%): Minimum market cap change to report
- **Email styling**: Modern HTML templates in `send_email_alert()` function

### System Configuration
- **Cache duration** (default: 24 hours): API response caching period
- **Schedule timing**: Cron expression in `daily_check.yml` (default: 6 PM UTC weekdays)
- **Archive retention** (default: 14 days): How long to keep analysis artifacts
- **API limits**: Market cap fetching limits (default: top 15 holdings)

### Data Validation Settings
- **Maximum weight threshold** (default: 25%): Cap for unrealistic weights
- **Reasonable changes limit** (default: 50): Threshold for suspicious bulk changes
- **Expected QQQ size** (90-110 components): Validation range for component count

## Troubleshooting

### ETF Data Issues

For QQQ data processing issues:

1. **Check Processing Logs**: GitHub Actions logs show detailed parsing attempts
2. **Review Raw Data**: Examine files in `data/raw/` directory via Actions artifacts
3. **Validation Reports**: Check the comprehensive data quality reports
4. **Use Analysis Tools**: Run `tools/etf_excel_converter.py` for manual inspection
5. **Verify Data Source**: Confirm Invesco QQQ URL is still accessible
6. **Check Fallback**: System automatically uses hardcoded data if download fails

### Email Notification Issues

- Check the Action logs in the GitHub Actions tab if notifications aren't being sent
- Ensure your email app password is correct and properly configured
- Check if the SMTP settings are accurate for your email provider

### Common Solutions

- **URL Changes**: Update Invesco URL in `download_qqq_holdings()` function if needed
- **API Rate Limits**: System includes 12-second delays and 24-hour caching
- **Parsing Failures**: Multiple fallback strategies handle format changes automatically
- **Email Issues**: Verify Gmail App Password and environment variable configuration
- **Data Quality**: Review validation reports to understand data characteristics
- **Missing Changes**: Check if changes are below configured thresholds

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.