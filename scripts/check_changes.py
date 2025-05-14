import os
import json
import time
import smtplib
import datetime
import random
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# Import required libraries at the module level
import pandas as pd
import yfinance as yf

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
SP500_FILE = DATA_DIR / "sp500_current.json"
QQQ_FILE = DATA_DIR / "qqq_current.json"
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")
TOP_POSITIONS_TO_TRACK = 20  # Track top 20 positions in each index
MAX_RETRIES = 3              # Maximum number of retries for API calls
RETRY_DELAY = 5              # Base delay between retries in seconds

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

def fetch_stock_data(ticker):
    """
    Fetch data for a single stock with retry logic.
    Returns stock information or None if failed after retries.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Add a small random delay to avoid hitting rate limits
            time.sleep(random.uniform(0.5, 1.5))
            
            stock = yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                # If rate limited, wait longer before retrying
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"Rate limited for {ticker}, waiting {wait_time}s before retry {attempt+1}/{MAX_RETRIES}")
                time.sleep(wait_time)
            else:
                print(f"Error fetching data for {ticker}: {e}")
                return None
    
    return None

def fetch_sp500_components():
    """
    Fetch current S&P 500 components and their market caps.
    Returns a list of dictionaries with symbol, name, and market cap.
    """
    print("Fetching S&P 500 components...")
    
    try:
        # Try to use existing data first to reduce API calls
        previous_data = load_previous_data(SP500_FILE)
        previous_by_symbol = {item["symbol"]: item for item in previous_data} if previous_data else {}
        
        # Use Wikipedia table as data source for component list
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        
        # Process top companies first to ensure we at least get data for them
        # Then process the rest - focus on the top positions
        top_companies = []
        other_companies = []
        
        # Get market cap data for each component, starting with top companies
        for idx, row in df.iterrows():
            ticker = row['Symbol']
            
            # Skip tickers with . or - as they often cause issues with yfinance
            if '.' in ticker or '-' in ticker:
                continue
                
            # Create basic component info
            component = {
                "symbol": ticker,
                "name": row['Security'],
                "market_cap": 0,
                "rank": 0
            }
            
            # Check if we have previous data we can use
            if ticker in previous_by_symbol:
                # Use market cap from previous data as fallback
                component["market_cap"] = previous_by_symbol[ticker]["market_cap"]
                
            # For top companies, always try to get fresh data
            if idx < TOP_POSITIONS_TO_TRACK * 3:  # Process more than we need for top positions
                top_companies.append(component)
            else:
                other_companies.append(component)
        
        # Get fresh data for top companies first
        for component in top_companies:
            ticker = component["symbol"]
            info = fetch_stock_data(ticker)
            if info:
                component["market_cap"] = info.get('marketCap', component["market_cap"])
        
        # Then process other companies 
        for component in other_companies:
            # Only fetch new data if we don't have market cap from previous data
            if component["market_cap"] == 0:
                ticker = component["symbol"]
                info = fetch_stock_data(ticker)
                if info:
                    component["market_cap"] = info.get('marketCap', 0)
        
        # Combine and sort all components
        components = top_companies + other_companies
        components.sort(key=lambda x: x["market_cap"], reverse=True)
        
        # Assign ranks
        for i, component in enumerate(components):
            component["rank"] = i + 1
        
        # If we have at least 30 components with market cap data, consider it successful
        if len([c for c in components if c["market_cap"] > 0]) > 30:
            return components
        else:
            print("WARNING: Not enough market cap data retrieved")
            return []
    except Exception as e:
        print(f"Error fetching S&P 500 components: {e}")
        return []

def fetch_qqq_components():
    """
    Fetch current QQQ (Nasdaq-100) components and their market caps.
    Returns a list of dictionaries with symbol, name, and market cap.
    """
    print("Fetching QQQ components...")
    
    try:
        # Try to use existing data first to reduce API calls
        previous_data = load_previous_data(QQQ_FILE)
        previous_by_symbol = {item["symbol"]: item for item in previous_data} if previous_data else {}
        
        # In a production environment, you should use a more reliable source
        # For this example, we'll use a predefined list of Nasdaq-100 tickers
        nasdaq100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 
            'AVGO', 'ADBE', 'COST', 'PEP', 'CSCO', 'NFLX', 'TMUS', 'CMCSA',
            'INTC', 'AMD', 'QCOM', 'INTU', 'TXN', 'AMGN', 'HON', 'AMAT',
            'SBUX', 'ADI', 'MDLZ', 'PYPL', 'REGN', 'GILD', 'LRCX', 'BKNG',
            'ADP', 'ISRG', 'VRTX', 'PANW', 'SNPS', 'CDNS', 'KLAC', 'ADSK'
        ]  # This is a partial list - in production you'd fetch the full list
        
        # Initialize components with previous data if available
        components = []
        for ticker in nasdaq100_tickers:
            component = {
                "symbol": ticker,
                "name": ticker,  # Default name
                "market_cap": 0,
                "rank": 0
            }
            
            # Use previous data if available
            if ticker in previous_by_symbol:
                component["name"] = previous_by_symbol[ticker]["name"]
                component["market_cap"] = previous_by_symbol[ticker]["market_cap"]
                
            components.append(component)
        
        # Process top companies first to ensure we at least get data for them
        for i, component in enumerate(components):
            # Only fetch fresh data for top companies or if we have no market cap data
            if i < TOP_POSITIONS_TO_TRACK * 2 or component["market_cap"] == 0:
                ticker = component["symbol"]
                info = fetch_stock_data(ticker)
                if info:
                    component["name"] = info.get('shortName', ticker)
                    component["market_cap"] = info.get('marketCap', component["market_cap"])
        
        # Sort by market cap and assign ranks
        components.sort(key=lambda x: x["market_cap"], reverse=True)
        for i, component in enumerate(components):
            component["rank"] = i + 1
        
        # If we have at least 15 components with market cap data, consider it successful
        if len([c for c in components if c["market_cap"] > 0]) > 15:
            return components
        else:
            print("WARNING: Not enough market cap data retrieved for QQQ")
            return []
    except Exception as e:
        print(f"Error fetching QQQ components: {e}")
        return []

def load_previous_data(file_path):
    """Load previous index data from JSON file."""
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error loading previous data from {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
    return []

def save_current_data(data, file_path):
    """Save current index data to JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

def detect_changes(previous_data, current_data):
    """
    Detect changes between previous and current index data.
    Returns a dictionary with different types of changes.
    """
    if not previous_data:
        return {"message": ["Initial data collection - no changes to report."], "top_changes": []}
    
    changes = {
        "message": [],      # All changes
        "top_changes": [],  # Changes in top positions
        "additions": [],    # New components
        "removals": [],     # Removed components
        "rank_changes": []  # Position changes
    }
    
    # Create dictionaries for easier comparison
    prev_by_symbol = {item["symbol"]: item for item in previous_data}
    curr_by_symbol = {item["symbol"]: item for item in current_data}
    
    # Check for additions/removals
    prev_symbols = set(prev_by_symbol.keys())
    curr_symbols = set(curr_by_symbol.keys())
    
    added = curr_symbols - prev_symbols
    removed = prev_symbols - curr_symbols
    
    for symbol in added:
        item = curr_by_symbol[symbol]
        message = f"ADDED: {item['name']} ({symbol}) at position #{item['rank']}"
        changes["message"].append(message)
        changes["additions"].append(message)
        
        # Check if in top positions
        if item['rank'] <= TOP_POSITIONS_TO_TRACK:
            changes["top_changes"].append(f"TOP {TOP_POSITIONS_TO_TRACK} ADDITION: {item['name']} ({symbol}) added at position #{item['rank']}")
    
    for symbol in removed:
        item = prev_by_symbol[symbol]
        message = f"REMOVED: {item['name']} ({symbol}) from position #{item['rank']}"
        changes["message"].append(message)
        changes["removals"].append(message)
        
        # Check if was in top positions
        if item['rank'] <= TOP_POSITIONS_TO_TRACK:
            changes["top_changes"].append(f"TOP {TOP_POSITIONS_TO_TRACK} REMOVAL: {item['name']} ({symbol}) removed from position #{item['rank']}")
    
    # Check for rank changes among existing components
    for symbol in prev_symbols & curr_symbols:
        prev_rank = prev_by_symbol[symbol]["rank"]
        curr_rank = curr_by_symbol[symbol]["rank"]
        
        if prev_rank != curr_rank:
            name = curr_by_symbol[symbol]["name"]
            direction = "up" if curr_rank < prev_rank else "down"
            places = abs(prev_rank - curr_rank)
            
            message = f"MOVED: {name} ({symbol}) from #{prev_rank} to #{curr_rank} ({direction} {places} place{'s' if places > 1 else ''})"
            changes["message"].append(message)
            changes["rank_changes"].append(message)
            
            # Special focus on top positions
            if prev_rank <= TOP_POSITIONS_TO_TRACK or curr_rank <= TOP_POSITIONS_TO_TRACK:
                changes["top_changes"].append(f"TOP {TOP_POSITIONS_TO_TRACK} MOVEMENT: {name} ({symbol}) moved from #{prev_rank} to #{curr_rank}")
    
    return changes

def send_email_alert(index_name, changes):
    """Send email alert with detected changes."""
    if not changes or not changes["message"] or changes["message"][0] == "Initial data collection - no changes to report.":
        print(f"No changes to report for {index_name}")
        return
    
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENT:
        print("Email configuration missing. Cannot send notification.")
        return
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = f"{index_name} Changes Detected - {today}"
    
    # Create HTML body with better formatting
    body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .top-change {{ background-color: #ffeb99; padding: 5px; margin: 2px 0; }}
            .regular-change {{ margin: 2px 0; }}
            h2 {{ color: #003366; }}
            h3 {{ color: #0066cc; }}
        </style>
    </head>
    <body>
        <h2>{index_name} Index Change Alert</h2>
        <p>Changes detected on {today}</p>
    """
    
    # Add top position changes with highlighting
    if changes["top_changes"]:
        body += f"<h3>Top {TOP_POSITIONS_TO_TRACK} Position Changes:</h3>\n<ul>"
        for change in changes["top_changes"]:
            body += f'<li class="top-change">{change}</li>\n'
        body += "</ul>"
    
    # Add other changes
    other_changes = [c for c in changes["message"] if c not in changes["top_changes"]]
    if other_changes:
        body += "<h3>Other Changes:</h3>\n<ul>"
        for change in other_changes:
            body += f'<li class="regular-change">{change}</li>\n'
        body += "</ul>"
    
    body += """
        <p>This alert was generated automatically by your Index Change Alert System.</p>
    </body>
    </html>
    """
    
    msg.attach(MIMEText(body, 'html'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"Email alert sent for {index_name} changes")
    except Exception as e:
        print(f"Error sending email: {e}")

def check_for_changes(index_name, data_file, fetch_function):
    """Check for changes in the index and return detected changes."""
    try:
        print(f"Checking for changes in {index_name}...")
        
        # Load previous data first - we need this regardless
        previous_data = load_previous_data(data_file)
        
        # Fetch current data
        current_data = fetch_function()
        
        # If this is the first run or we failed to get current data
        if not previous_data and not current_data:
            print(f"Initial run for {index_name} but couldn't fetch data. Will try again next time.")
            return {"message": ["Initial data collection - no data yet."], "top_changes": []}
        
        # If we have no previous data but got current data, save it for next time
        if not previous_data and current_data:
            print(f"Initial data collection for {index_name} - saving for future comparison")
            save_current_data(current_data, data_file)
            return {"message": ["Initial data collection - no changes to report."], "top_changes": []}
            
        # If we failed to get current data but have previous data
        if not current_data and previous_data:
            print(f"WARNING: Failed to fetch current {index_name} data, will use previous data")
            return {"message": [f"Unable to fetch current {index_name} data. Using previous data."], "top_changes": []}
        
        # Detect changes
        changes = detect_changes(previous_data, current_data)
        
        # Log changes if found
        if changes["message"] and changes["message"][0] != "Initial data collection - no changes to report.":
            print(f"{index_name} changes detected: {len(changes['message'])}")
            print(f"Top {TOP_POSITIONS_TO_TRACK} changes: {len(changes['top_changes'])}")
            
            for change in changes["top_changes"]:
                print(f"  - {change}")
            
            for change in changes["message"]:
                if change not in changes["top_changes"]:
                    print(f"  - {change}")
        else:
            print(f"No {index_name} changes detected or initial data collection")
        
        # Save current data for next time (only if we successfully fetched it)
        if current_data:
            save_current_data(current_data, data_file)
        
        return changes
        
    except Exception as e:
        print(f"Error checking for changes in {index_name}: {e}")
        return {"message": [], "top_changes": []}

def main():
    """Main function to check for index changes and send alerts."""
    print("Starting index change check...")
    
    try:
        # Check S&P 500
        sp500_changes = check_for_changes("S&P 500", SP500_FILE, fetch_sp500_components)
        send_email_alert("S&P 500", sp500_changes)
        
        # Check QQQ
        qqq_changes = check_for_changes("QQQ", QQQ_FILE, fetch_qqq_components)
        send_email_alert("QQQ", qqq_changes)
        
        print("Index change check completed successfully")
        
    except Exception as e:
        print(f"Critical error in main function: {e}")

if __name__ == "__main__":
    main()