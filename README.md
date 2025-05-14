# S&P 500 and QQQ Index Change Alert System

This repository contains a GitHub Action that automatically tracks changes in the S&P 500 and QQQ (Nasdaq-100) indices. When changes occur, especially in key positions like the #10 rank, the system sends email notifications with details about the changes.

## Features

- Daily checks of S&P 500 and QQQ index compositions
- Detects additions, removals, and rank changes
- Special focus on the #10 position in indices
- Email notifications with detailed change information
- Completely free to run (uses GitHub Actions)

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
2. It fetches the current composition of both indices
3. Compares with previously stored data to detect changes
4. If changes are found, especially in the #10 position, it sends an email alert
5. Updates the stored data for the next comparison

## Customization

- Modify `check_changes.py` to track different index positions or add more indices
- Adjust the cron schedule in `daily_check.yml` to run at different times
- Change the email format or add more notification methods

## Troubleshooting

- Check the Action logs in the GitHub Actions tab if notifications aren't being sent
- Ensure your email app password is correct and properly configured
- For Gmail, make sure "Less secure app access" is enabled or use an app password
  
