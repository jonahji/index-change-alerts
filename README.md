# S&P 500 and QQQ Index Change Alert System

This repository contains a GitHub Action that automatically tracks changes in the S&P 500 and QQQ (Nasdaq-100) indices using official ETF provider data. When changes occur, especially in the top 20 positions, the system sends email notifications with detailed information.

## Features

- Daily checks of S&P 500 and QQQ index compositions using **official ETF holdings data**
- Uses State Street (SPY) and Invesco (QQQ) ETF data as the source of truth
- Detects additions, removals, rank changes, and significant weight changes
- Special focus on the top 20 positions in both indices
- **Enhanced reporting of market cap changes with percentage tracking**
- **Smart caching to reduce API calls and improve reliability**
- **Robust ETF file parsing** with multiple fallback mechanisms
- **Raw data preservation** for troubleshooting
- Improved email notifications with categorized and highlighted changes
- Completely free to run (uses GitHub Actions)

## Why This Approach?

This system uses official ETF holdings data directly from the fund providers, which has several advantages:
- More reliable than third-party financial APIs
- No rate limiting issues or API key requirements for the core data
- Contains accurate weight information directly from the ETF providers
- Includes official ranking of components

## Setup Instructions

1. **Fork or clone this repository**

2. **Configure GitHub Secrets**

   Add the following secrets to your repository:
   
   - `EMAIL_SENDER`: Email address that will send notifications (e.g., your Gmail)
   - `EMAIL_APP_PASSWORD`: App password for your email (for Gmail, create one at https://myaccount.google.com/apppasswords)
   - `EMAIL_RECIPIENT`: Email address where you want to receive the notifications
   - `ALPHA_VANTAGE_API_KEY` (Optional): For enhanced market cap data on top stocks

3. **Initial Run**

   After setting up secrets, manually trigger the workflow by going to the "Actions" tab in your repository, selecting "Daily Index Change Check" workflow, and clicking "Run workflow".

   This will create the initial data files. Future runs will compare against these files to detect changes.

4. **Automatic Execution**

   The workflow is scheduled to run automatically at 6:00 PM UTC (after market close) on weekdays.

## How It Works

1. The GitHub Action runs on the defined schedule
2. It downloads the latest holdings data from State Street (SPY) and Invesco (QQQ) ETF providers
3. Processes the data using robust parsing with multiple fallback mechanisms
4. Enhances the data with real-time market cap information for top stocks (optional)
5. Compares with previously stored data to detect changes in composition, ranks, and weights
6. If changes are found, especially in the top 20 positions, it sends an email alert
7. Updates the stored data for the next comparison

## Repository Structure

```
index-change-alerts/
├── .github/workflows/        # GitHub Actions configuration
│   └── daily_check.yml
├── data/                     # Stored index compositions and cache
│   ├── sp500_current.json
│   ├── qqq_current.json
│   ├── cache/                # Cached API responses
│   └── raw/                  # Raw downloaded ETF files
├── scripts/                  # Python scripts
│   └── check_changes.py      # Main script with robust ETF integration
├── tools/                    # Local development tools
│   └── etf_excel_converter.py  # Excel file analyzer and converter
├── notebooks/                # Jupyter notebooks for development
│   └── etf_holdings_analyzer.py  # ETF data exploration
└── README.md                 # Documentation
```

## ETF Provider Data Integration

The system integrates with official ETF provider data sources:

### SPY (S&P 500 ETF) - State Street Global Advisors
- Data URL: https://www.ssga.com/us/en/individual/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx
- Update frequency: Daily (after market close)
- Format: Excel file (.xlsx)

### QQQ (Nasdaq-100 ETF) - Invesco
- Data URL: https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ
- Update frequency: Daily (after market close)
- Format: Excel file (.xlsx)

## Robust ETF File Processing

The system uses multiple approaches to handle ETF provider data:

1. **Multiple Parsing Methods**: Tries different row skipping and Excel engines
2. **Intelligent Column Detection**: Identifies key columns regardless of naming variations
3. **Proper Weight Formatting**: Handles various weight formats (percentages, decimals)
4. **Raw File Preservation**: Keeps original downloads for troubleshooting
5. **Fallback Mechanism**: Uses hardcoded data if downloads or parsing fails

## Local Development Tools

The repository includes tools for local development and troubleshooting:

### ETF Excel Converter

A utility to analyze and convert ETF provider Excel files:

```bash
# Inspect an Excel file
python tools/etf_excel_converter.py --mode inspect --input data/raw/spy_holdings_raw.xlsx

# Convert to CSV for easier viewing
python tools/etf_excel_converter.py --mode convert --input data/raw/spy_holdings_raw.xlsx

# Batch process all raw files
python tools/etf_excel_converter.py --mode batch --input data/raw
```

### ETF Holdings Analyzer

A script to download and explore ETF holdings data:

```bash
# Run the analyzer to download and process ETF holdings
python notebooks/etf_holdings_analyzer.py
```

## Customization

You can easily customize the system by:

- Changing `TOP_POSITIONS_TO_TRACK` (currently 20) to focus on more or fewer positions
- Modifying the email format in the `send_email_alert()` function
- Adjusting the threshold for significant weight changes (currently 0.1%)
- Changing the caching duration in the `cache_api_response()` function
- Changing the cron schedule in `daily_check.yml` to run at different times

## Troubleshooting

### ETF Data Issues

If you encounter issues with ETF data downloads:

1. Check the GitHub Actions logs for specific error messages
2. Examine the raw Excel files stored in the `data/raw` directory
3. Use the ETF Excel Converter tool to analyze file structure
4. Verify that the ETF provider URLs haven't changed

### Email Notification Issues

- Check the Action logs in the GitHub Actions tab if notifications aren't being sent
- Ensure your email app password is correct and properly configured
- Check if the SMTP settings are accurate for your email provider

### Other Common Issues

- If the ETF provider URLs change, update them in the `download_spy_holdings()` and `download_qqq_holdings()` functions
- If API rate limits are exceeded, adjust the `fetch_top_stocks_data()` function to use fewer requests or longer delays

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.