import os
import json
import datetime
import smtplib
import requests
import time
import pandas as pd
import io
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
SP500_FILE = DATA_DIR / "sp500_current.json"
QQQ_FILE = DATA_DIR / "qqq_current.json"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"  # Directory to store raw downloaded files
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") or os.environ.get("EMAIL_APP_PASSWORD")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")
TOP_POSITIONS_TO_TRACK = 20  # Track top 20 positions in each index

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

def cache_api_response(symbol, data, cache_duration_hours=24):
    """Cache API response data for a symbol."""
    try:
        cache_file = CACHE_DIR / f"{symbol}_api_cache.json"
        
        cache_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "expires": (datetime.datetime.now() + datetime.timedelta(hours=cache_duration_hours)).isoformat(),
            "data": data
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
            
        print(f"Cached API data for {symbol}")
    except Exception as e:
        print(f"Error caching API data for {symbol}: {e}")

def get_cached_api_response(symbol):
    """Get cached API response if valid."""
    try:
        cache_file = CACHE_DIR / f"{symbol}_api_cache.json"
        
        if not cache_file.exists():
            return None
            
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            
        # Check if cache is expired
        expires = datetime.datetime.fromisoformat(cache_data["expires"])
        if datetime.datetime.now() > expires:
            print(f"Cache for {symbol} is expired")
            return None
            
        print(f"Using cached API data for {symbol}")
        return cache_data["data"]
    except Exception as e:
        print(f"Error reading cached API data for {symbol}: {e}")
        return None

def fetch_real_data(symbol, api_key=None, use_cache=True):
    """
    Fetch real market cap data for a stock using Alpha Vantage API.
    Uses caching to reduce API calls.
    """
    # Check cache first if enabled
    if use_cache:
        cached_data = get_cached_api_response(symbol)
        if cached_data:
            return cached_data
    
    if api_key is None:
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
        
    try:
        # Use Alpha Vantage API (free tier)
        # Limited to 5 API calls per minute and 500 per day
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
        
        print(f"Fetching real data for {symbol}...")
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return None
            
        data = response.json()
        
        # Check if we got valid data
        if "MarketCapitalization" not in data:
            print(f"Warning: No market cap data found for {symbol}")
            if "Note" in data:
                print(f"API message: {data['Note']}")
            return None
            
        market_cap = float(data.get("MarketCapitalization", 0))
        name = data.get("Name", f"{symbol} Inc.")
        
        print(f"Successfully fetched real data for {symbol}: {name}, Market Cap: ${market_cap/1e9:.2f}B")
        
        result = {
            "symbol": symbol,
            "name": name,
            "market_cap": market_cap,
            "real_data": True
        }
        
        # Cache the result
        if use_cache:
            cache_api_response(symbol, result)
        
        return result
    except Exception as e:
        print(f"Error fetching real data for {symbol}: {e}")
        return None

def fetch_top_stocks_data(symbols, max_stocks=10):
    """
    Fetch real data for multiple stocks, respecting API rate limits.
    Returns a dictionary of stock data by symbol.
    """
    if not symbols:
        return {}
        
    # Limit the number of API calls to respect free tier limits
    symbols_to_fetch = symbols[:max_stocks]
    print(f"Fetching real data for top {len(symbols_to_fetch)} stocks: {', '.join(symbols_to_fetch)}")
    
    results = {}
    for i, symbol in enumerate(symbols_to_fetch):
        # Check if we already have cached data
        cached_data = get_cached_api_response(symbol)
        if cached_data:
            results[symbol] = cached_data
            continue
            
        # Fetch data for this symbol
        stock_data = fetch_real_data(symbol, use_cache=False)  # Don't check cache again
        
        if stock_data:
            results[symbol] = stock_data
            
        # Add delay to respect rate limits (5 calls per minute = 12 seconds between calls)
        if i < len(symbols_to_fetch) - 1:  # Don't wait after the last request
            seconds_to_wait = 12
            print(f"Waiting {seconds_to_wait} seconds before next API call (respecting rate limits)...")
            time.sleep(seconds_to_wait)
    
    print(f"Successfully fetched data for {len(results)} out of {len(symbols_to_fetch)} stocks")
    return results

def download_spy_holdings():
    """
    Download the latest holdings for SPY (S&P 500 ETF) from State Street.
    Returns a list of holdings in standardized format.
    
    Improved to handle various Excel file formats and save raw files.
    """
    # State Street SPDR (SPY) holdings URL
    url = "https://www.ssga.com/us/en/individual/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
    
    print(f"Downloading SPY holdings from: {url}")
    
    try:
        # Download the Excel file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the raw file for inspection
        today = datetime.datetime.now().strftime("%Y%m%d")
        raw_file = RAW_DIR / f"spy_holdings_raw_{today}.xlsx"
        with open(raw_file, 'wb') as f:
            f.write(response.content)
        print(f"Saved raw SPY holdings to {raw_file}")
        
        # Try different parsing approaches
        result = None
        
        # Approach 1: Standard parsing with skiprows=3 (typical format)
        try:
            df = pd.read_excel(io.BytesIO(response.content), skiprows=3)
            
            # Check if we got sensible data by looking for expected columns
            if 'Ticker' in df.columns and any(col for col in df.columns if 'Weight' in col):
                print("Successfully parsed SPY holdings with standard approach")
                
                # Extract key columns
                ticker_col = 'Ticker'
                name_col = 'Security Description' if 'Security Description' in df.columns else None
                weight_col = next((col for col in df.columns if 'Weight' in col), None)
                
                # Basic cleanup
                df = df.dropna(subset=[ticker_col])
                
                # Format data into standardized structure
                result = []
                for i, row in df.iterrows():
                    try:
                        symbol = str(row[ticker_col]).strip() if pd.notna(row[ticker_col]) else ""
                        if not symbol:  # Skip rows without a valid ticker
                            continue
                            
                        # Format the data
                        item = {
                            "symbol": symbol,
                            "rank": i + 1  # 1-based rank
                        }
                        
                        # Add name if available
                        if name_col and pd.notna(row[name_col]):
                            item["name"] = str(row[name_col]).strip()
                        else:
                            item["name"] = f"{symbol} Inc."
                        
                        # Add weight if available - handle potential formatting issues
                        if weight_col and pd.notna(row[weight_col]):
                            weight_val = row[weight_col]
                            if isinstance(weight_val, str):
                                # Try to extract numeric value if it's a string
                                try:
                                    # Remove % signs, commas
                                    weight_val = weight_val.replace('%', '').replace(',', '')
                                    item["weight"] = float(weight_val)
                                except:
                                    # Use regex as fallback
                                    match = re.search(r'(\d+\.?\d*)', weight_val)
                                    if match:
                                        item["weight"] = float(match.group(1))
                            else:
                                # Numeric value
                                item["weight"] = float(weight_val)
                                
                            # If weight is not in percentage form (0-100), convert it
                            if "weight" in item and item["weight"] < 1.0:
                                item["weight"] = item["weight"] * 100
                        
                        result.append(item)
                    except Exception as e:
                        print(f"Error processing SPY row {i}: {e}")
                        continue
                
                print(f"Processed {len(result)} SPY holdings with standard approach")
            else:
                print("Standard parsing approach failed - missing expected columns")
        except Exception as e:
            print(f"Error with standard parsing approach: {e}")
        
        # Approach 2: Try different skiprows values if standard approach failed
        if not result:
            for skiprows in [0, 1, 2, 4, 5]:
                try:
                    print(f"Trying alternative parsing with skiprows={skiprows}")
                    df = pd.read_excel(io.BytesIO(response.content), skiprows=skiprows)
                    
                    # Look for key columns
                    ticker_col = None
                    name_col = None
                    weight_col = None
                    
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'ticker' in col_lower or 'symbol' in col_lower:
                            ticker_col = col
                        elif 'name' in col_lower or 'description' in col_lower or 'security' in col_lower:
                            name_col = col
                        elif 'weight' in col_lower or 'wt.' in col_lower or 'wt' in col_lower:
                            weight_col = col
                    
                    if ticker_col and weight_col:
                        print(f"Found key columns: Ticker={ticker_col}, Name={name_col}, Weight={weight_col}")
                        # Similar processing as above...
                        df = df.dropna(subset=[ticker_col])
                        
                        result = []
                        for i, row in df.iterrows():
                            try:
                                symbol = str(row[ticker_col]).strip() if pd.notna(row[ticker_col]) else ""
                                if not symbol:
                                    continue
                                    
                                item = {
                                    "symbol": symbol,
                                    "rank": i + 1
                                }
                                
                                if name_col and pd.notna(row[name_col]):
                                    item["name"] = str(row[name_col]).strip()
                                else:
                                    item["name"] = f"{symbol} Inc."
                                
                                if weight_col and pd.notna(row[weight_col]):
                                    weight_val = row[weight_col]
                                    if isinstance(weight_val, str):
                                        try:
                                            weight_val = weight_val.replace('%', '').replace(',', '')
                                            item["weight"] = float(weight_val)
                                        except:
                                            match = re.search(r'(\d+\.?\d*)', weight_val)
                                            if match:
                                                item["weight"] = float(match.group(1))
                                    else:
                                        item["weight"] = float(weight_val)
                                        
                                    if "weight" in item and item["weight"] < 1.0:
                                        item["weight"] = item["weight"] * 100
                                
                                result.append(item)
                            except Exception as e:
                                print(f"Error processing row {i} with alternative approach: {e}")
                                continue
                        
                        print(f"Processed {len(result)} SPY holdings with alternative approach (skiprows={skiprows})")
                        break
                except Exception as e:
                    print(f"Alternative parsing approach (skiprows={skiprows}) failed: {e}")
        
        # Approach 3: Try using openpyxl engine if other approaches failed
        if not result:
            try:
                print("Trying parsing with openpyxl engine")
                df = pd.read_excel(io.BytesIO(response.content), engine='openpyxl')
                
                # Similar column detection and processing...
                ticker_col = None
                name_col = None
                weight_col = None
                
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'ticker' in col_lower or 'symbol' in col_lower:
                        ticker_col = col
                    elif 'name' in col_lower or 'description' in col_lower or 'security' in col_lower:
                        name_col = col
                    elif 'weight' in col_lower or 'wt.' in col_lower or 'wt' in col_lower:
                        weight_col = col
                
                if ticker_col and weight_col:
                    print(f"Found key columns with openpyxl engine: Ticker={ticker_col}, Name={name_col}, Weight={weight_col}")
                    
                    # Process data similar to above approaches
                    df = df.dropna(subset=[ticker_col])
                    
                    result = []
                    for i, row in df.iterrows():
                        try:
                            symbol = str(row[ticker_col]).strip() if pd.notna(row[ticker_col]) else ""
                            if not symbol:
                                continue
                                
                            item = {
                                "symbol": symbol,
                                "rank": i + 1
                            }
                            
                            if name_col and pd.notna(row[name_col]):
                                item["name"] = str(row[name_col]).strip()
                            else:
                                item["name"] = f"{symbol} Inc."
                            
                            if weight_col and pd.notna(row[weight_col]):
                                weight_val = row[weight_col]
                                if isinstance(weight_val, str):
                                    try:
                                        weight_val = weight_val.replace('%', '').replace(',', '')
                                        item["weight"] = float(weight_val)
                                    except:
                                        match = re.search(r'(\d+\.?\d*)', weight_val)
                                        if match:
                                            item["weight"] = float(match.group(1))
                                else:
                                    item["weight"] = float(weight_val)
                                    
                                if "weight" in item and item["weight"] < 1.0:
                                    item["weight"] = item["weight"] * 100
                            
                            result.append(item)
                        except Exception as e:
                            print(f"Error processing row {i} with openpyxl engine: {e}")
                            continue
                    
                    print(f"Processed {len(result)} SPY holdings with openpyxl engine")
            except Exception as e:
                print(f"Parsing with openpyxl engine failed: {e}")
        
        # If we have successfully parsed data, return it
        if result and len(result) > 0:
            print(f"Successfully processed {len(result)} SPY holdings")
            return result
        else:
            print("All parsing attempts failed for SPY holdings")
            # Fallback to hardcoded data
            print("Falling back to hardcoded S&P 500 data...")
            return get_sp500_components_hardcoded()
    
    except Exception as e:
        print(f"Error downloading SPY holdings: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to hardcoded data if download fails
        print("Falling back to hardcoded S&P 500 data...")
        return get_sp500_components_hardcoded()

def download_qqq_holdings():
    """
    Download the latest holdings for QQQ (Nasdaq-100 ETF) from Invesco.
    Returns a list of holdings in standardized format.
    
    Improved to handle various Excel file formats and save raw files.
    """
    # Invesco QQQ holdings URL
    url = "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ"
    
    print(f"Downloading QQQ holdings from: {url}")
    
    try:
        # Download the Excel file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the raw file for inspection
        today = datetime.datetime.now().strftime("%Y%m%d")
        raw_file = RAW_DIR / f"qqq_holdings_raw_{today}.xlsx"
        with open(raw_file, 'wb') as f:
            f.write(response.content)
        print(f"Saved raw QQQ holdings to {raw_file}")
        
        # Try different parsing approaches
        result = None
        
        # Approach 1: Standard parsing with skiprows=1 (typical format)
        try:
            df = pd.read_excel(io.BytesIO(response.content), skiprows=1)
            
            # Check for expected columns
            expected_cols = ['Holding Ticker', 'Holding Name', 'Weight']
            if all(col in df.columns for col in expected_cols):
                print("Successfully parsed QQQ holdings with standard approach")
                
                # Extract key columns
                ticker_col = 'Holding Ticker'
                name_col = 'Holding Name'
                weight_col = 'Weight'
                
                # Basic cleanup
                df = df.dropna(subset=[ticker_col])
                
                # Format data into standardized structure
                result = []
                for i, row in df.iterrows():
                    try:
                        symbol = str(row[ticker_col]).strip() if pd.notna(row[ticker_col]) else ""
                        if not symbol:  # Skip rows without a valid ticker
                            continue
                            
                        # Format the data
                        item = {
                            "symbol": symbol,
                            "rank": i + 1  # 1-based rank
                        }
                        
                        # Add name if available
                        if name_col and pd.notna(row[name_col]):
                            item["name"] = str(row[name_col]).strip()
                        else:
                            item["name"] = f"{symbol} Inc."
                        
                        # Add weight if available - handle potential formatting issues
                        if weight_col and pd.notna(row[weight_col]):
                            weight_val = row[weight_col]
                            if isinstance(weight_val, str):
                                # Try to extract numeric value if it's a string
                                try:
                                    # Remove % signs, commas
                                    weight_val = weight_val.replace('%', '').replace(',', '')
                                    item["weight"] = float(weight_val)
                                except:
                                    # Use regex as fallback
                                    match = re.search(r'(\d+\.?\d*)', weight_val)
                                    if match:
                                        item["weight"] = float(match.group(1))
                            else:
                                # Numeric value
                                item["weight"] = float(weight_val)
                                
                            # If weight is not in percentage form (0-100), convert it
                            if "weight" in item and item["weight"] < 1.0:
                                item["weight"] = item["weight"] * 100
                        
                        result.append(item)
                    except Exception as e:
                        print(f"Error processing QQQ row {i}: {e}")
                        continue
                
                print(f"Processed {len(result)} QQQ holdings with standard approach")
            else:
                print("Standard parsing approach failed - missing expected columns")
                print(f"Found columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error with standard parsing approach: {e}")
        
        # Approach 2: Try different skiprows values if standard approach failed
        if not result:
            for skiprows in [0, 2, 3, 4]:
                try:
                    print(f"Trying alternative parsing with skiprows={skiprows}")
                    df = pd.read_excel(io.BytesIO(response.content), skiprows=skiprows)
                    
                    # Look for key columns
                    ticker_col = None
                    name_col = None
                    weight_col = None
                    
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'ticker' in col_lower or 'symbol' in col_lower:
                            ticker_col = col
                        elif 'name' in col_lower or 'description' in col_lower or 'security' in col_lower:
                            name_col = col
                        elif 'weight' in col_lower or 'wt.' in col_lower or 'wt' in col_lower:
                            weight_col = col
                    
                    if ticker_col and weight_col:
                        print(f"Found key columns: Ticker={ticker_col}, Name={name_col}, Weight={weight_col}")
                        # Similar processing as above...
                        df = df.dropna(subset=[ticker_col])
                        
                        result = []
                        for i, row in df.iterrows():
                            try:
                                symbol = str(row[ticker_col]).strip() if pd.notna(row[ticker_col]) else ""
                                if not symbol:
                                    continue
                                    
                                item = {
                                    "symbol": symbol,
                                    "rank": i + 1
                                }
                                
                                if name_col and pd.notna(row[name_col]):
                                    item["name"] = str(row[name_col]).strip()
                                else:
                                    item["name"] = f"{symbol} Inc."
                                
                                if weight_col and pd.notna(row[weight_col]):
                                    weight_val = row[weight_col]
                                    if isinstance(weight_val, str):
                                        try:
                                            weight_val = weight_val.replace('%', '').replace(',', '')
                                            item["weight"] = float(weight_val)
                                        except:
                                            match = re.search(r'(\d+\.?\d*)', weight_val)
                                            if match:
                                                item["weight"] = float(match.group(1))
                                    else:
                                        item["weight"] = float(weight_val)
                                        
                                    if "weight" in item and item["weight"] < 1.0:
                                        item["weight"] = item["weight"] * 100
                                
                                result.append(item)
                            except Exception as e:
                                print(f"Error processing row {i} with alternative approach: {e}")
                                continue
                        
                        print(f"Processed {len(result)} QQQ holdings with alternative approach (skiprows={skiprows})")
                        break
                except Exception as e:
                    print(f"Alternative parsing approach (skiprows={skiprows}) failed: {e}")
        
        # Approach 3: Try using openpyxl engine if other approaches failed
        if not result:
            try:
                print("Trying parsing with openpyxl engine")
                df = pd.read_excel(io.BytesIO(response.content), engine='openpyxl')
                
                # Similar column detection and processing...
                ticker_col = None
                name_col = None
                weight_col = None
                
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'ticker' in col_lower or 'symbol' in col_lower:
                        ticker_col = col
                    elif 'name' in col_lower or 'description' in col_lower or 'security' in col_lower:
                        name_col = col
                    elif 'weight' in col_lower or 'wt.' in col_lower or 'wt' in col_lower:
                        weight_col = col
                
                if ticker_col and weight_col:
                    print(f"Found key columns with openpyxl engine: Ticker={ticker_col}, Name={name_col}, Weight={weight_col}")
                    
                    # Process data similar to above approaches
                    df = df.dropna(subset=[ticker_col])
                    
                    result = []
                    for i, row in df.iterrows():
                        try:
                            symbol = str(row[ticker_col]).strip() if pd.notna(row[ticker_col]) else ""
                            if not symbol:
                                continue
                                
                            item = {
                                "symbol": symbol,
                                "rank": i + 1
                            }
                            
                            if name_col and pd.notna(row[name_col]):
                                item["name"] = str(row[name_col]).strip()
                            else:
                                item["name"] = f"{symbol} Inc."
                            
                            if weight_col and pd.notna(row[weight_col]):
                                weight_val = row[weight_col]
                                if isinstance(weight_val, str):
                                    try:
                                        weight_val = weight_val.replace('%', '').replace(',', '')
                                        item["weight"] = float(weight_val)
                                    except:
                                        match = re.search(r'(\d+\.?\d*)', weight_val)
                                        if match:
                                            item["weight"] = float(match.group(1))
                                else:
                                    item["weight"] = float(weight_val)
                                    
                                if "weight" in item and item["weight"] < 1.0:
                                    item["weight"] = item["weight"] * 100
                            
                            result.append(item)
                        except Exception as e:
                            print(f"Error processing row {i} with openpyxl engine: {e}")
                            continue
                    
                    print(f"Processed {len(result)} QQQ holdings with openpyxl engine")
            except Exception as e:
                print(f"Parsing with openpyxl engine failed: {e}")
        
        # If we have successfully parsed data, return it
        if result and len(result) > 0:
            print(f"Successfully processed {len(result)} QQQ holdings")
            return result
        else:
            print("All parsing attempts failed for QQQ holdings")
            # Fallback to hardcoded data
            print("Falling back to hardcoded QQQ data...")
            return get_qqq_components_hardcoded()
    
    except Exception as e:
        print(f"Error downloading QQQ holdings: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to hardcoded data if download fails
        print("Falling back to hardcoded QQQ data...")
        return get_qqq_components_hardcoded()

def get_sp500_components_hardcoded():
    """
    Hardcoded S&P 500 components as fallback.
    """
    print("Using hardcoded S&P 500 top components...")
    
    # Top 20 S&P 500 components with approximate weights
    sp500_components = [
        {"symbol": "AAPL", "name": "Apple Inc.", "weight": 7.24, "rank": 1},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "weight": 6.85, "rank": 2},
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "weight": 5.01, "rank": 3},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "weight": 3.59, "rank": 4},
        {"symbol": "META", "name": "Meta Platforms Inc. Class A", "weight": 2.34, "rank": 5},
        {"symbol": "GOOG", "name": "Alphabet Inc. Class C", "weight": 1.98, "rank": 6},
        {"symbol": "GOOGL", "name": "Alphabet Inc. Class A", "weight": 1.71, "rank": 7},
        {"symbol": "BRK.B", "name": "Berkshire Hathaway Inc. Class B", "weight": 1.69, "rank": 8},
        {"symbol": "TSLA", "name": "Tesla, Inc.", "weight": 1.67, "rank": 9},
        {"symbol": "AVGO", "name": "Broadcom Inc.", "weight": 1.34, "rank": 10},
        {"symbol": "UNH", "name": "UnitedHealth Group Incorporated", "weight": 1.32, "rank": 11},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "weight": 1.22, "rank": 12},
        {"symbol": "XOM", "name": "Exxon Mobil Corporation", "weight": 1.19, "rank": 13},
        {"symbol": "LLY", "name": "Eli Lilly and Company", "weight": 1.16, "rank": 14},
        {"symbol": "V", "name": "Visa Inc. Class A", "weight": 1.10, "rank": 15},
        {"symbol": "COST", "name": "Costco Wholesale Corporation", "weight": 1.04, "rank": 16},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "weight": 1.02, "rank": 17},
        {"symbol": "MA", "name": "Mastercard Incorporated Class A", "weight": 0.94, "rank": 18},
        {"symbol": "PG", "name": "Procter & Gamble Company", "weight": 0.91, "rank": 19},
        {"symbol": "HD", "name": "Home Depot, Inc.", "weight": 0.86, "rank": 20}
    ]
    return sp500_components

def get_qqq_components_hardcoded():
    """
    Hardcoded QQQ components as fallback.
    """
    print("Using hardcoded QQQ top components...")
    
    # Top 20 QQQ components with approximate weights
    qqq_components = [
        {"symbol": "AAPL", "name": "Apple Inc.", "weight": 11.94, "rank": 1},
        {"symbol": "MSFT", "name": "Microsoft Corp.", "weight": 10.20, "rank": 2},
        {"symbol": "NVDA", "name": "NVIDIA Corp.", "weight": 7.52, "rank": 3},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "weight": 6.82, "rank": 4},
        {"symbol": "META", "name": "Meta Platforms Inc. Class A", "weight": 4.99, "rank": 5},
        {"symbol": "TSLA", "name": "Tesla Inc.", "weight": 3.53, "rank": 6},
        {"symbol": "GOOG", "name": "Alphabet Inc. Class C", "weight": 3.49, "rank": 7},
        {"symbol": "GOOGL", "name": "Alphabet Inc. Class A", "weight": 3.15, "rank": 8},
        {"symbol": "AVGO", "name": "Broadcom Inc.", "weight": 2.79, "rank": 9},
        {"symbol": "COST", "name": "Costco Wholesale Corp.", "weight": 2.21, "rank": 10},
        {"symbol": "CSCO", "name": "Cisco Systems Inc.", "weight": 1.95, "rank": 11},
        {"symbol": "ADBE", "name": "Adobe Inc.", "weight": 1.92, "rank": 12},
        {"symbol": "TMUS", "name": "T-Mobile US Inc.", "weight": 1.87, "rank": 13},
        {"symbol": "AMD", "name": "Advanced Micro Devices Inc.", "weight": 1.49, "rank": 14},
        {"symbol": "PEP", "name": "PepsiCo Inc.", "weight": 1.47, "rank": 15},
        {"symbol": "NFLX", "name": "Netflix Inc.", "weight": 1.38, "rank": 16},
        {"symbol": "CMCSA", "name": "Comcast Corp. Class A", "weight": 1.37, "rank": 17},
        {"symbol": "INTU", "name": "Intuit Inc.", "weight": 1.33, "rank": 18},
        {"symbol": "QCOM", "name": "Qualcomm Inc.", "weight": 1.19, "rank": 19},
        {"symbol": "TXN", "name": "Texas Instruments Inc.", "weight": 1.15, "rank": 20}
    ]
    return qqq_components

def get_sp500_components():
    """
    Get S&P 500 components with real data for the top stocks.
    Now tries to download from official source first with improved parsing.
    """
    print("Getting S&P 500 components...")
    
    try:
        # First, try to download from official ETF provider source
        components = download_spy_holdings()
        
        if not components:
            # Fallback to hardcoded data
            components = get_sp500_components_hardcoded()
        
        # Use top components list for market cap data
        # Extract symbols of top stocks
        top_symbols = [component["symbol"] for component in components[:10]]
        
        # Fetch real data for top stocks
        real_data = fetch_top_stocks_data(top_symbols, max_stocks=10)
        
        # Update components with real data
        for i, component in enumerate(components):
            symbol = component["symbol"]
            if symbol in real_data:
                # Update with real data but keep rank and weight
                updated_data = real_data[symbol].copy()
                updated_data["rank"] = component["rank"]
                updated_data["weight"] = component["weight"]
                components[i] = updated_data
                print(f"Updated {symbol} with real market cap data")
        
        return components
    except Exception as e:
        print(f"Error getting S&P 500 components: {e}")
        # Final fallback
        return get_sp500_components_hardcoded()

def get_qqq_components():
    """
    Get QQQ (Nasdaq-100) components with real data for the top stocks.
    Now tries to download from official source first with improved parsing.
    """
    print("Getting QQQ components...")
    
    try:
        # First, try to download from official ETF provider source
        components = download_qqq_holdings()
        
        if not components:
            # Fallback to hardcoded data
            components = get_qqq_components_hardcoded()
        
        # Use top components list for market cap data
        # Extract symbols of top stocks
        top_symbols = [component["symbol"] for component in components[:10]]
        
        # Fetch real data for top stocks
        real_data = fetch_top_stocks_data(top_symbols, max_stocks=10)
        
        # Update components with real data
        for i, component in enumerate(components):
            symbol = component["symbol"]
            if symbol in real_data:
                # Update with real data but keep rank and weight
                updated_data = real_data[symbol].copy()
                updated_data["rank"] = component["rank"]
                updated_data["weight"] = component["weight"]
                components[i] = updated_data
                print(f"Updated {symbol} with real market cap data")
        
        return components
    except Exception as e:
        print(f"Error getting QQQ components: {e}")
        # Final fallback
        return get_qqq_components_hardcoded()

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
        return {"message": ["Initial data collection - no changes to report."], 
                "top_changes": [],
                "additions": [],
                "removals": [],
                "rank_changes": [],
                "weight_changes": [],
                "market_cap_changes": []}
    
    changes = {
        "message": [],            # All changes
        "top_changes": [],        # Changes in top positions
        "additions": [],          # New components
        "removals": [],           # Removed components
        "rank_changes": [],       # Position changes
        "weight_changes": [],     # Weight changes
        "market_cap_changes": []  # Market cap changes
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
        weight = item.get('weight', 0)
        message = f"ADDED: {item['name']} ({symbol}) at position #{item['rank']}"
        if 'weight' in item:
            message += f" with weight {weight:.2f}%"
        changes["message"].append(message)
        changes["additions"].append(message)
        
        # Check if in top positions
        if item['rank'] <= TOP_POSITIONS_TO_TRACK:
            changes["top_changes"].append(f"TOP {TOP_POSITIONS_TO_TRACK} ADDITION: {item['name']} ({symbol}) added at position #{item['rank']}")
    
    for symbol in removed:
        item = prev_by_symbol[symbol]
        weight = item.get('weight', 0)
        message = f"REMOVED: {item['name']} ({symbol}) from position #{item['rank']}"
        if 'weight' in item:
            message += f" (previous weight {weight:.2f}%)"
        changes["message"].append(message)
        changes["removals"].append(message)
        
        # Check if was in top positions
        if item['rank'] <= TOP_POSITIONS_TO_TRACK:
            changes["top_changes"].append(f"TOP {TOP_POSITIONS_TO_TRACK} REMOVAL: {item['name']} ({symbol}) removed from position #{item['rank']}")
    
    # Check for rank changes and weight changes among existing components
    for symbol in prev_symbols & curr_symbols:
        prev_rank = prev_by_symbol[symbol]["rank"]
        curr_rank = curr_by_symbol[symbol]["rank"]
        
        # Get weights if they exist
        prev_weight = prev_by_symbol[symbol].get("weight", 0)
        curr_weight = curr_by_symbol[symbol].get("weight", 0)
        
        # Check for significant weight changes (more than 0.1 percentage point)
        # Only if both previous and current data have weight information
        if 'weight' in prev_by_symbol[symbol] and 'weight' in curr_by_symbol[symbol]:
            weight_change = curr_weight - prev_weight
            if abs(weight_change) > 0.1:  # More than 0.1 percentage point
                name = curr_by_symbol[symbol]["name"]
                weight_msg = f"Weight changed from {prev_weight:.2f}% to {curr_weight:.2f}% ({weight_change:+.2f}%)"
                weight_change_msg = f"WEIGHT CHANGE: {name} ({symbol}) - {weight_msg}"
                changes["message"].append(weight_change_msg)
                changes["weight_changes"].append(weight_change_msg)
                
                # Special focus on significant weight changes in top positions
                if prev_rank <= TOP_POSITIONS_TO_TRACK or curr_rank <= TOP_POSITIONS_TO_TRACK:
                    if abs(weight_change) > 0.5:  # More significant threshold for top positions
                        changes["top_changes"].append(f"TOP {TOP_POSITIONS_TO_TRACK} SIGNIFICANT WEIGHT CHANGE: {name} ({symbol}) {weight_msg}")
        
        # Check for market cap changes if available
        if 'market_cap' in prev_by_symbol[symbol] and 'market_cap' in curr_by_symbol[symbol]:
            prev_mcap = prev_by_symbol[symbol]["market_cap"]
            curr_mcap = curr_by_symbol[symbol]["market_cap"]
            
            # Calculate percentage change
            mcap_pct_change = ((curr_mcap - prev_mcap) / prev_mcap) * 100
            
            # Report significant market cap changes (more than 5%)
            if abs(mcap_pct_change) > 5:
                name = curr_by_symbol[symbol]["name"]
                mcap_msg = f"Market cap changed from ${prev_mcap/1e9:.2f}B to ${curr_mcap/1e9:.2f}B ({mcap_pct_change:+.2f}%)"
                mcap_change_msg = f"MARKET CAP CHANGE: {name} ({symbol}) - {mcap_msg}"
                changes["message"].append(mcap_change_msg)
                changes["market_cap_changes"].append(mcap_change_msg)
                
                # Special focus on significant market cap changes in top positions
                if prev_rank <= TOP_POSITIONS_TO_TRACK or curr_rank <= TOP_POSITIONS_TO_TRACK:
                    if abs(mcap_pct_change) > 10:  # More significant threshold for top positions
                        changes["top_changes"].append(f"TOP {TOP_POSITIONS_TO_TRACK} SIGNIFICANT MARKET CAP CHANGE: {name} ({symbol}) {mcap_msg}")
        
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
    
    # Check email configuration
    if not EMAIL_SENDER:
        print("Email configuration missing: EMAIL_SENDER environment variable not set")
        return
    if not EMAIL_PASSWORD:
        print("Email configuration missing: EMAIL_PASSWORD or EMAIL_APP_PASSWORD environment variable not set")
        return
    if not EMAIL_RECIPIENT:
        print("Email configuration missing: EMAIL_RECIPIENT environment variable not set")
        return
        
    print(f"Preparing email alert for {index_name} changes...")
    print(f"Sender: {EMAIL_SENDER}")
    print(f"Recipient: {EMAIL_RECIPIENT}")
    
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
        body += f"<h3>⭐ Top {TOP_POSITIONS_TO_TRACK} Position Changes:</h3>\n<ul>"
        for change in changes["top_changes"]:
            body += f'<li class="top-change">{change}</li>\n'
        body += "</ul>"
    
    # Add other changes by category
    # Rank changes
    rank_changes = [c for c in changes["rank_changes"] if c not in [tc for tc in changes["top_changes"] if "MOVEMENT" in tc]]
    if rank_changes:
        body += "<h3>📊 Rank Changes:</h3>\n<ul>"
        for change in rank_changes:
            body += f'<li class="regular-change">{change}</li>\n'
        body += "</ul>"
    
    # Weight changes    
    weight_changes = [c for c in changes["weight_changes"] if c not in [tc for tc in changes["top_changes"] if "WEIGHT" in tc]]
    if weight_changes:
        body += "<h3>⚖️ Weight Changes:</h3>\n<ul>"
        for change in weight_changes:
            body += f'<li class="regular-change">{change}</li>\n'
        body += "</ul>"
    
    # Market cap changes
    market_cap_changes = [c for c in changes["market_cap_changes"] if c not in [tc for tc in changes["top_changes"] if "MARKET CAP" in tc]]
    if market_cap_changes:
        body += "<h3>💰 Market Cap Changes:</h3>\n<ul>"
        for change in market_cap_changes:
            body += f'<li class="regular-change">{change}</li>\n'
        body += "</ul>"
    
    # Additions
    additions = [c for c in changes["additions"] if c not in [tc for tc in changes["top_changes"] if "ADDITION" in tc]]
    if additions:
        body += "<h3>➕ Additions:</h3>\n<ul>"
        for change in additions:
            body += f'<li class="regular-change">{change}</li>\n'
        body += "</ul>"
    
    # Removals
    removals = [c for c in changes["removals"] if c not in [tc for tc in changes["top_changes"] if "REMOVAL" in tc]]
    if removals:
        body += "<h3>➖ Removals:</h3>\n<ul>"
        for change in removals:
            body += f'<li class="regular-change">{change}</li>\n'
        body += "</ul>"
    
    # Other changes not covered in specific categories
    other_changes = [c for c in changes["message"] 
                    if c not in changes["top_changes"] 
                    and c not in rank_changes
                    and c not in weight_changes
                    and c not in market_cap_changes
                    and c not in changes["additions"] 
                    and c not in changes["removals"]]
    if other_changes:
        body += "<h3>ℹ️ Other Changes:</h3>\n<ul>"
        for change in other_changes:
            body += f'<li class="regular-change">{change}</li>\n'
        body += "</ul>"
    
    body += f"""
        <p>This alert was generated automatically by your Index Change Alert System using official ETF provider data.</p>
        <p><small>Data source: {'State Street SPY ETF' if index_name == 'S&P 500' else 'Invesco QQQ ETF'}</small></p>
    </body>
    </html>
    """
    
    msg.attach(MIMEText(body, 'html'))
    
    try:
        print("Connecting to Gmail SMTP server...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        print("Logging in to Gmail...")
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        print("Sending email...")
        server.send_message(msg)
        server.quit()
        print(f"Email alert sent for {index_name} changes")
    except Exception as e:
        print(f"Error sending email: {e}")
        print("\nTroubleshooting Gmail authentication issues:")
        print("1. Make sure you're using an App Password for EMAIL_PASSWORD, not your regular Gmail password")
        print("2. Generate an App Password at: https://myaccount.google.com/apppasswords")
        print("3. In GitHub repository settings, update the EMAIL_APP_PASSWORD secret with the new App Password")

def check_for_changes(index_name, data_file, get_components_function):
    """Check for changes in the index and return detected changes."""
    try:
        print(f"Checking for changes in {index_name}...")
        
        # Load previous data
        previous_data = load_previous_data(data_file)
        print(f"Loaded previous data: {len(previous_data) if previous_data else 0} items")
        
        # Get current components 
        print(f"Getting current data for {index_name}...")
        current_data = get_components_function()
        print(f"Got current data: {len(current_data)} items")
        
        # If this is the first run
        if not previous_data and current_data:
            print(f"Initial data collection for {index_name} - saving for future comparison")
            save_current_data(current_data, data_file)
            return {"message": ["Initial data collection - no changes to report."], "top_changes": []}
        
        print(f"Detecting changes for {index_name}...")
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
        print(f"Saving current data for {index_name}...")
        save_current_data(current_data, data_file)
        print(f"Data saved to {data_file}")
        
        return changes
        
    except Exception as e:
        print(f"Error checking for changes in {index_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"message": [], "top_changes": []}

def main():
    """Main function to check for index changes and send alerts."""
    print("Starting index change check...")
    print(f"Data directory: {DATA_DIR} (Exists: {DATA_DIR.exists()})")
    
    try:
        # Verify directories exist
        DATA_DIR.mkdir(exist_ok=True)
        CACHE_DIR.mkdir(exist_ok=True)
        RAW_DIR.mkdir(exist_ok=True)
        print(f"Created/verified data directory: {DATA_DIR}")
        print(f"Created/verified cache directory: {CACHE_DIR}")
        print(f"Created/verified raw files directory: {RAW_DIR}")
        
        # Check S&P 500
        sp500_changes = check_for_changes("S&P 500", SP500_FILE, get_sp500_components)
        send_email_alert("S&P 500", sp500_changes)
        
        # Check QQQ
        qqq_changes = check_for_changes("QQQ", QQQ_FILE, get_qqq_components)
        send_email_alert("QQQ", qqq_changes)
        
        print("Index change check completed successfully")
        
    except Exception as e:
        print(f"Critical error in main function: {e}")

if __name__ == "__main__":
    main()