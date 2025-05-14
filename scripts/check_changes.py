import os
import json
import smtplib
import datetime
import requests
import yfinance as yf
import pandas as pd
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
TOP_POSITIONS_TO_TRACK = 20  # Track top 20 positions in each index

class IndexTracker:
    """Class to track changes in stock market indices."""
    
    def __init__(self, index_name, data_file):
        self.index_name = index_name
        self.data_file = data_file
    
    def fetch_components(self):
        """Fetch current components of the index."""
        if self.index_name == "S&P 500":
            return self._fetch_sp500_components()
        elif self.index_name == "QQQ":
            return self._fetch_qqq_components()
        else:
            raise ValueError(f"Unsupported index: {self.index_name}")
    
    def _fetch_sp500_components(self):
        """Fetch S&P 500 components and market caps."""
        print(f"Fetching {self.index_name} components...")
        
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
            print(f"Error fetching {self.index_name} components: {e}")
            return []
    
    def _fetch_qqq_components(self):
        """Fetch QQQ (Nasdaq-100) components and market caps."""
        print(f"Fetching {self.index_name} components...")
        
        try:
            # In a production environment, you should use a more reliable source
            # For this example, we'll use a simplified approach with yfinance
            
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
            print(f"Error fetching {self.index_name} components: {e}")
            return []
    
    def load_previous_data(self):
        """Load previous index data from JSON file."""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_current_data(self, data):
        """Save current index data to JSON file."""
        DATA_DIR.mkdir(exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def detect_changes(self, previous_data, current_data):
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
    
    def check_for_changes(self):
        """Check for changes in the index and return detected changes."""
        current_data = self.fetch_components()
        previous_data = self.load_previous_data()
        
        changes = self.detect_changes(previous_data, current_data)
        
        # Log changes if found
        if changes["message"] and changes["message"][0] != "Initial data collection - no changes to report.":
            print(f"{self.index_name} changes detected: {len(changes['message'])}")
            print(f"Top {TOP_POSITIONS_TO_TRACK} changes: {len(changes['top_changes'])}")
            
            for change in changes["top_changes"]:
                print(f"  - {change}")
            
            for change in changes["message"]:
                if change not in changes["top_changes"]:
                    print(f"  - {change}")
        else:
            print(f"No {self.index_name} changes detected")
        
        # Save current data for next time
        if current_data:
            self.save_current_data(current_data)
        
        return changes


class EmailNotifier:
    """Class to send email notifications about index changes."""
    
    def __init__(self, sender, password, recipient):
        self.sender = sender
        self.password = password
        self.recipient = recipient
    
    def send_alert(self, index_name, changes):
        """Send email alert with detected changes."""
        if not changes["message"] or changes["message"][0] == "Initial data collection - no changes to report.":
            return
        
        if not self.sender or not self.password or not self.recipient:
            print("Email configuration missing. Cannot send notification.")
            return
        
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        msg = MIMEMultipart()
        msg['From'] = self.sender
        msg['To'] = self.recipient
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
            server.login(self.sender, self.password)
            server.send_message(msg)
            server.quit()
            print(f"Email alert sent for {index_name} changes")
        except Exception as e:
            print(f"Error sending email: {e}")


def main():
    """Main function to check for index changes and send alerts."""
    print("Starting index change check...")
    
    # Initialize notifier
    notifier = EmailNotifier(
        sender=EMAIL_SENDER,
        password=EMAIL_PASSWORD,
        recipient=EMAIL_RECIPIENT
    )
    
    # Check S&P 500
    sp500_tracker = IndexTracker("S&P 500", SP500_FILE)
    sp500_changes = sp500_tracker.check_for_changes()
    notifier.send_alert("S&P 500", sp500_changes)
    
    # Check QQQ
    qqq_tracker = IndexTracker("QQQ", QQQ_FILE)
    qqq_changes = qqq_tracker.check_for_changes()
    notifier.send_alert("QQQ", qqq_changes)
    
    print("Index change check completed")


if __name__ == "__main__":
    main()