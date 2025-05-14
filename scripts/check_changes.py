import os
import json
import time
import smtplib
import datetime
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import pandas as pd

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
SP500_FILE = DATA_DIR / "sp500_current.json"
QQQ_FILE = DATA_DIR / "qqq_current.json"
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")
TOP_POSITIONS_TO_TRACK = 20  # Track top 20 positions in each index

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

def fetch_spy_holdings():
    """
    Fetch current SPY ETF holdings from State Street (S&P 500).
    Returns a list of dictionaries with symbol, name, and weight.
    """
    print("Fetching SPY (S&P 500) holdings...")
    
    try:
        # State Street publishes holdings data in CSV format
        url = "https://www.ssga.com/us/en/individual/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
        
        # Download the Excel file
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to download SPY holdings. Status code: {response.status_code}")
            return []
            
        # Save the file temporarily
        temp_file = DATA_DIR / "spy_holdings_temp.xlsx"
        with open(temp_file, "wb") as f:
            f.write(response.content)
        
        # Read the Excel file using pandas
        df = pd.read_excel(temp_file, sheet_name=0, skiprows=3)
        
        # Clean up
        try:
            os.remove(temp_file)
        except:
            pass
        
        # Process holdings data
        holdings = []
        for _, row in df.iterrows():
            try:
                # Column names may change slightly over time - adjust as needed
                ticker = str(row.get('Ticker') or "")
                name = str(row.get('Name') or "")
                weight = float(row.get('Weight') or 0)
                
                # Skip cash components and non-stock entries
                if not ticker or ticker.lower() == 'cash' or ticker.lower() == 'nan':
                    continue
                    
                holdings.append({
                    "symbol": ticker,
                    "name": name,
                    "weight": weight,
                    "rank": 0  # Will fill in later
                })
            except Exception as e:
                print(f"Error processing SPY holding row: {e}")
        
        # Sort by weight and assign ranks
        holdings.sort(key=lambda x: x["weight"], reverse=True)
        for i, holding in enumerate(holdings):
            holding["rank"] = i + 1
            
        return holdings
    except Exception as e:
        print(f"Error fetching SPY holdings: {e}")
        return []

def fetch_qqq_holdings():
    """
    Fetch current QQQ ETF holdings from Invesco (Nasdaq-100).
    Returns a list of dictionaries with symbol, name, and weight.
    """
    print("Fetching QQQ (Nasdaq-100) holdings...")
    
    try:
        # Invesco publishes holdings data in CSV format
        url = "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ"
        
        # Download the CSV file
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to download QQQ holdings. Status code: {response.status_code}")
            return []
            
        # Parse CSV data
        lines = response.text.splitlines()
        
        # Find the data section (skipping headers)
        start_idx = 0
        for i, line in enumerate(lines):
            if "Ticker" in line and "Weight" in line:
                start_idx = i + 1
                break
        
        # Process holdings data
        holdings = []
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if not line.strip():
                continue  # Skip empty lines
                
            parts = line.split(',')
            if len(parts) < 3:
                continue  # Skip malformed lines
                
            try:
                ticker = parts[0].strip().replace('"', '')
                name = parts[1].strip().replace('"', '')
                weight_str = parts[-1].strip().replace('"', '').replace('%', '')
                
                # Handle blank or invalid tickers
                if not ticker or ticker.lower() == 'cash':
                    continue
                    
                weight = float(weight_str) if weight_str else 0
                
                holdings.append({
                    "symbol": ticker,
                    "name": name,
                    "weight": weight,
                    "rank": 0  # Will fill in later
                })
            except Exception as e:
                print(f"Error processing QQQ holding line: {e}")
        
        # Sort by weight and assign ranks
        holdings.sort(key=lambda x: x["weight"], reverse=True)
        for i, holding in enumerate(holdings):
            holding["rank"] = i + 1
            
        return holdings
    except Exception as e:
        print(f"Error fetching QQQ holdings: {e}")
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
        message = f"ADDED: {item['name']} ({symbol}) at position #{item['rank']} with weight {item['weight']:.2f}%"
        changes["message"].append(message)
        changes["additions"].append(message)
        
        # Check if in top positions
        if item['rank'] <= TOP_POSITIONS_TO_TRACK:
            changes["top_changes"].append(f"TOP {TOP_POSITIONS_TO_TRACK} ADDITION: {item['name']} ({symbol}) added at position #{item['rank']}")
    
    for symbol in removed:
        item = prev_by_symbol[symbol]
        message = f"REMOVED: {item['name']} ({symbol}) from position #{item['rank']} (previous weight {item['weight']:.2f}%)"
        changes["message"].append(message)
        changes["removals"].append(message)
        
        # Check if was in top positions
        if item['rank'] <= TOP_POSITIONS_TO_TRACK:
            changes["top_changes"].append(f"TOP {TOP_POSITIONS_TO_TRACK} REMOVAL: {item['name']} ({symbol}) removed from position #{item['rank']}")
    
    # Check for rank changes among existing components
    for symbol in prev_symbols & curr_symbols:
        prev_rank = prev_by_symbol[symbol]["rank"]
        curr_rank = curr_by_symbol[symbol]["rank"]
        prev_weight = prev_by_symbol[symbol]["weight"]
        curr_weight = curr_by_symbol[symbol]["weight"]
        
        # Check for significant weight changes (more than 0.1 percentage point)
        weight_change = curr_weight - prev_weight
        if abs(weight_change) > 0.1:
            name = curr_by_symbol[symbol]["name"]
            weight_msg = f"Weight changed from {prev_weight:.2f}% to {curr_weight:.2f}% ({weight_change:+.2f}%)"
            changes["message"].append(f"WEIGHT CHANGE: {name} ({symbol}) - {weight_msg}")
        
        # Check for rank changes
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
        
        # Load previous data
        previous_data = load_previous_data(data_file)
        
        # Fetch current data
        current_data = fetch_function()
        
        # If this is the first run or we failed to get current data
        if not current_data:
            print(f"WARNING: Failed to fetch current {index_name} data")
            return {"message": [], "top_changes": []}
            
        # If we have no previous data but got current data, save it for next time
        if not previous_data and current_data:
            print(f"Initial data collection for {index_name} - saving for future comparison")
            save_current_data(current_data, data_file)
            return {"message": ["Initial data collection - no changes to report."], "top_changes": []}
        
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
        
        # Save current data for next time
        save_current_data(current_data, data_file)
        
        return changes
        
    except Exception as e:
        print(f"Error checking for changes in {index_name}: {e}")
        return {"message": [], "top_changes": []}

def main():
    """Main function to check for index changes and send alerts."""
    print("Starting index change check...")
    
    try:
        # Check S&P 500 (SPY holdings)
        sp500_changes = check_for_changes("S&P 500", SP500_FILE, fetch_spy_holdings)
        send_email_alert("S&P 500", sp500_changes)
        
        # Check QQQ (Nasdaq-100)
        qqq_changes = check_for_changes("QQQ", QQQ_FILE, fetch_qqq_holdings)
        send_email_alert("QQQ", qqq_changes)
        
        print("Index change check completed successfully")
        
    except Exception as e:
        print(f"Critical error in main function: {e}")

if __name__ == "__main__":
    main()