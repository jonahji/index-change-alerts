# S&P 500 and QQQ Index Change Alert System

This repository contains a GitHub Action that automatically tracks changes in the S&P 500 and QQQ (Nasdaq-100) indices using official ETF provider data. When changes occur, especially in the top 20 positions, the system sends email notifications with detailed information.

## Features

- Daily checks of S&P 500 and QQQ index compositions using official ETF holdings data
- Uses State Street (SPY) and Invesco (QQQ) ETF data as the source of truth
- Detects additions, removals, rank changes, and significant weight changes
- Special focus on the top 20 positions in both indices
- Enhanced email notifications with highlighted important changes
- Completely free to run (uses GitHub Actions)

## Why This Approach?

This system uses official ETF holdings data directly from the fund providers, which has several advantages:
- More reliable than third-party financial APIs
- No rate limiting issues
- Contains accurate weight information
- Includes official ranking of components

## Setup Instructions

1. **Fork or clone this repository**

2. **Configure GitHub Secrets**

   Add the following secrets to your repository:
   
   - `EMAIL_SENDER`: Email address that will send notifications (e.g., your Gmail)
   - `EMAIL_APP_PASSWORD`: App password for your email (for Gmail, create one at https://myaccount.google.com/apppasswords)
   - `EMAIL_RECIPIENT`: Email address where you want to receive the notifications

3. **Initial Run**

   After setting up secrets, manually trigger the workflow by going to the "Actions" tab in your repository, selecting "Daily Index Change Check" workflow, and clicking "Run workflow".

   This will create the initial data files. Future runs will compare against these files to detect changes.

4. **Automatic Execution**

   The workflow is scheduled to run automatically at 6:00 PM UTC (after US market close) on weekdays.

## How It Works

1. The GitHub Action runs on the defined schedule
2. It downloads the latest holdings data from State Street (SPY) and Invesco (QQQ)
3. Compares with previously stored data to detect changes in composition and weights
4. If changes are found, especially in the top 20 positions, it sends an email alert
5. Updates the stored data for the next comparison

## Customization

You can easily customize the system by:

- Changing `TOP_POSITIONS_TO_TRACK` (currently 20) to focus on more or fewer positions
- Modifying the email format in the `send_email_alert()` function
- Adjusting the threshold for significant weight changes (currently 0.1%)
- Changing the cron schedule in `daily_check.yml` to run at different times

## Troubleshooting

- Check the Action logs in the GitHub Actions tab if notifications aren't being sent
- Ensure your email app password is correct and properly configured
- If the ETF provider URLs change, you may need to update the URLs in the script