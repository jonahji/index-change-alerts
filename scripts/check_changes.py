import os
import json
import smtplib
import datetime
import requests
import yfinance as yf
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
SP500_FILE = DATA_DIR / "sp500_current.json"
QQQ_FILE = DATA_DIR / "qqq_current.json"
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")

def fetch_sp500_components():
    """
    Fetch current S&P 500 components and their market caps.
    Returns a list of dictionaries with symbol, name, and market cap.
    """
    print("Fetching S&P 500 components...")
    
    try:
        # Use Wikipedia table as data source
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        
        # Get market cap data for each component
        components = []
        for _, row in df.iterrows():
            ticker = row['Symbol']
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', 0)
                components.append({
                    "symbol": ticker,
                    "name": row['Security'],
                    "market_cap": market_cap,
                    "rank": 0  # Will fill in later
                })
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        # Sort by market cap and assign ranks
        components.sort(key=lambda x: x["market_cap"], reverse=True)
        for i, component in enumerate(components):
            component["rank"] = i + 1
            
        return components
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
        # Fetch QQQ holdings from Invesco website
        # Note: In practice, you'd need to parse this from the Invesco site
        # For this demo, we'll use a simplified approach with yfinance
        
        nasdaq100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 
            'AVGO', 'ADBE', 'COST', 'PEP', 'CSCO', 'NFLX', 'TMUS', 'CMCSA',
            'INTC', 'AMD', 'QCOM', 'INTU', 'TXN', 'AMGN', 'HON', 'AMAT',
            'SBUX', 'ADI', 'MDLZ', 'PYPL', 'REGN', 'GILD', 'LRCX', 'BKNG',
            'ADP', 'ISRG', 'VRTX', 'PANW', 'SNPS', 'CDNS', 'KLAC', 'ADSK'
        ]  # This is a partial list - in production you'd fetch the full list
        
        components = []
        for ticker in nasdaq100_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get('marketCap', 0)
                components.append({
                    "symbol": ticker,
                    "name": info.get('shortName', ticker),
                    "market_cap": market_cap,
                    "rank": 0  # Will fill in later
                })
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        # Sort by market cap and assign ranks
        components.sort(key=lambda x: x["market_cap"], reverse=True)
        for i, component in enumerate(components):
            component["rank"] = i + 1
            
        return components
    except Exception as e:
        print(f"Error fetching QQQ components: {e}")
        return []

def load_previous_data(file_path):
    """Load previous index data from JSON file."""
    if file_path.exists():
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

def save_current_data(data, file_path):
    """Save current index data to JSON file."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def detect_changes(previous_data, current_data):
    """
    Detect changes between previous and current index data.
    Returns a list of change descriptions.
    """
    if not previous_data:
        return ["Initial data collection - no changes to report."]
    
    changes = []
    
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
        changes.append(f"ADDED: {item['name']} ({symbol}) at position #{item['rank']}")
    
    for symbol in removed:
        item = prev_by_symbol[symbol]
        changes.append(f"REMOVED: {item['name']} ({symbol}) from position #{item['rank']}")
    
    # Check for rank changes among existing components
    for symbol in prev_symbols & curr_symbols:
        prev_rank = prev_by_symbol[symbol]["rank"]
        curr_rank = curr_by_symbol[symbol]["rank"]
        
        if prev_rank != curr_rank:
            name = curr_by_symbol[symbol]["name"]
            changes.append(f"MOVED: {name} ({symbol}) from #{prev_rank} to #{curr_rank}")
            
            # Special focus on the #10 position
            if prev_rank == 10 or curr_rank == 10:
                changes.append(f"IMPORTANT CHANGE at #10 position: {name} ({symbol})")
    
    return changes

def send_email_alert(index_name, changes):
    """Send email alert with detected changes."""
    if not changes or not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENT:
        return
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = f"{index_name} Changes Detected - {today}"
    
    body = f"<h2>{index_name} Index Change Alert</h2>\n<h3>Changes detected on {today}:</h3>\n<ul>"
    for change in changes:
        body += f"<li>{change}</li>\n"
    body += "</ul>"
    
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

def main():
    """Main function to check for index changes and send alerts."""
    # Import pandas here to avoid installation if not running the script
    import pandas as pd
    
    print("Starting index change check...")
    
    # Check S&P 500
    sp500_current = fetch_sp500_components()
    sp500_previous = load_previous_data(SP500_FILE)
    
    sp500_changes = detect_changes(sp500_previous, sp500_current)
    if sp500_changes and sp500_changes[0] != "Initial data collection - no changes to report.":
        print(f"S&P 500 changes detected: {len(sp500_changes)}")
        for change in sp500_changes:
            print(f"  - {change}")
        send_email_alert("S&P 500", sp500_changes)
    else:
        print("No S&P 500 changes detected")
    
    # Save current S&P 500 data
    if sp500_current:
        save_current_data(sp500_current, SP500_FILE)
    
    # Check QQQ
    qqq_current = fetch_qqq_components()
    qqq_previous = load_previous_data(QQQ_FILE)
    
    qqq_changes = detect_changes(qqq_previous, qqq_current)
    if qqq_changes and qqq_changes[0] != "Initial data collection - no changes to report.":
        print(f"QQQ changes detected: {len(qqq_changes)}")
        for change in qqq_changes:
            print(f"  - {change}")
        send_email_alert("QQQ", qqq_changes)
    else:
        print("No QQQ changes detected")
    
    # Save current QQQ data
    if qqq_current:
        save_current_data(qqq_current, QQQ_FILE)
    
    print("Index change check completed")

if __name__ == "__main__":
    main()