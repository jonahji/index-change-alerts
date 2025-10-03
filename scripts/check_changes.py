import os
import json
import datetime
import smtplib
import requests
import time
import pandas as pd
import io
import re
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
QQQ_FILE = DATA_DIR / "qqq_current.json"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"  # Directory to store raw downloaded files
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD") or os.environ.get("EMAIL_APP_PASSWORD")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")
TOP_POSITIONS_TO_TRACK = 20  # Track top 20 positions in each index
MAX_WEIGHT_THRESHOLD = 25.0  # Maximum realistic weight for an index component (%)
MAX_REASONABLE_CHANGES = 50  # Maximum number of changes before flagging as suspicious

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

def infer_weight_format(weights):
    """
    Infer whether weight values are already in percentage form or need conversion.
    
    Args:
        weights: List of weight values
    
    Returns:
        (needs_conversion, explanation): Tuple with boolean indicating if values need to be multiplied by 100
                                         and a string explaining the reasoning
    """
    if not weights:
        return False, "No weight values to analyze"
    
    # Calculate basic statistics
    weights_array = np.array(weights)
    total_weight = np.sum(weights_array)
    max_weight = np.max(weights_array)
    
    # Case 1: If the total weight is already close to 100, values are likely percentages
    if 80 <= total_weight <= 120:
        return False, f"Total weight ({total_weight:.2f}) is close to 100%, suggesting values are already percentages"
    
    # Case 2: If the total weight is very small (<10), values are likely decimals needing conversion
    if total_weight < 10:
        return True, f"Total weight ({total_weight:.2f}) is very small, suggesting decimal values need conversion to percentages"
    
    # Case 3: If the maximum value is large (>50), values are likely already percentages
    if max_weight > 25:
        return False, f"Maximum weight ({max_weight:.2f}) exceeds typical ETF component weight, suggesting values are already percentages"
    
    # Case 4: If we have many small values (<1.0), but total is significant, they're likely already percentages
    small_values = weights_array < 1.0
    percent_small = np.mean(small_values) * 100
    if percent_small > 70:  # If more than 70% of values are <1.0
        return False, f"{percent_small:.1f}% of weights are <1.0, suggesting values are already small percentages"
    
    # Default case: assume needed conversion if most values are very small
    return (np.mean(weights_array) < 0.5), f"Mean weight ({np.mean(weights_array):.4f}) suggests {'conversion needed' if np.mean(weights_array) < 0.5 else 'no conversion needed'}"

def normalize_weights(weights, component_names=None):
    """
    Enhanced weight normalization with robust error handling and validation.

    Args:
        weights: List of weight values (can contain None, NaN, strings)
        component_names: Optional list of component names for logging

    Returns:
        List of normalized weight values in percentage form
    """
    if not weights:
        print("ℹ️  No weight data provided for normalization")
        return []

    try:
        # Safe conversion to numeric array
        weights_array = pd.to_numeric(weights, errors='coerce').values
        names = component_names if component_names else [f"Component {i+1}" for i in range(len(weights))]

        # Enhanced validation: Check for invalid values
        invalid_mask = pd.isna(weights_array) | np.isinf(weights_array) | (weights_array < 0)
        if np.any(invalid_mask):
            invalid_count = np.sum(invalid_mask)
            print(f"⚠️  WARNING: Found {invalid_count} invalid weight values (NaN, inf, or negative)")
            # Replace invalid values with 0 and log them
            invalid_indices = np.where(invalid_mask)[0]
            for idx in invalid_indices[:5]:  # Log first 5 invalid values
                if idx < len(names):
                    print(f"     Invalid weight for {names[idx]}: {weights[idx]}")
            if len(invalid_indices) > 5:
                print(f"     ... and {len(invalid_indices) - 5} more invalid values")
            weights_array = np.where(invalid_mask, 0, weights_array)

    except Exception as e:
        print(f"❌ Error in weight array conversion: {e}")
        print(f"   Input weights sample: {weights[:5] if len(weights) > 5 else weights}")
        return []

    # Check if conversion needed
    needs_conversion, explanation = infer_weight_format(weights_array)
    print(f"✓ Weight format analysis: {explanation}")

    # Apply conversion if needed
    if needs_conversion:
        print("✓ Converting weight values from decimal to percentage (multiplying by 100)")
        normalized = weights_array * 100
    else:
        print("✓ Weight values appear to already be in percentage format")
        normalized = weights_array

    # Enhanced validation after normalization
    total_weight = np.sum(normalized)
    print(f"✓ Total weight sum: {total_weight:.2f}%")

    if total_weight < 80 or total_weight > 120:
        print(f"⚠️  WARNING: Total weight ({total_weight:.2f}%) is outside expected range (80-120%)")
        print("   This may indicate data quality issues or missing components")

    # Check for unreasonable individual values
    max_value = np.max(normalized)
    if max_value > MAX_WEIGHT_THRESHOLD:
        if needs_conversion:
            print(f"⚠️  WARNING: After conversion, maximum weight ({max_value:.2f}%) exceeds threshold ({MAX_WEIGHT_THRESHOLD}%).")
            print("   This suggests the weight format detection might be incorrect. Reverting conversion.")
            normalized = weights_array  # Revert to original values
        else:
            print(f"⚠️  WARNING: Maximum weight ({max_value:.2f}%) exceeds threshold ({MAX_WEIGHT_THRESHOLD}%).")
            idx = np.argmax(normalized)
            print(f"   Highest weight is for {names[idx]}: {normalized[idx]:.2f}%")

            # Cap extreme values with better logging
            extreme_mask = normalized > MAX_WEIGHT_THRESHOLD
            if np.any(extreme_mask):
                extreme_indices = np.where(extreme_mask)[0]
                print(f"   Capping {len(extreme_indices)} extreme weight values:")
                for idx in extreme_indices:
                    print(f"     - {names[idx]}: {normalized[idx]:.2f}% → {MAX_WEIGHT_THRESHOLD}%")
                    normalized[idx] = MAX_WEIGHT_THRESHOLD

    # Final validation summary
    final_total = np.sum(normalized)
    valid_weights = np.sum(normalized > 0)
    print(f"✓ Validation complete: {valid_weights} valid weights, total = {final_total:.2f}%")

    return normalized.tolist()

def validate_qqq_data(components):
    """
    Comprehensive validation for QQQ component data with robust null handling.

    Args:
        components: List of component dictionaries (may contain None values)

    Returns:
        (is_valid, validation_report): Tuple with boolean and detailed report
    """
    if not components:
        return False, "❌ No components data provided"

    # Filter out None/invalid components
    valid_components = [comp for comp in components if comp and isinstance(comp, dict)]
    if len(valid_components) != len(components):
        invalid_count = len(components) - len(valid_components)
        print(f"⚠️  Filtered out {invalid_count} invalid/None components")
        components = valid_components

    if not components:
        return False, "❌ No valid components after filtering"

    report = []
    warnings = []
    errors = []

    # Enhanced structure validation with null safety
    required_fields = ['symbol', 'name', 'rank']
    for i, comp in enumerate(components):
        if not isinstance(comp, dict):
            errors.append(f"Component {i+1}: Invalid data type (expected dict, got {type(comp)})")
            continue

        missing_fields = []
        invalid_fields = []

        for field in required_fields:
            if field not in comp:
                missing_fields.append(field)
            elif comp[field] is None or (isinstance(comp[field], str) and not comp[field].strip()):
                invalid_fields.append(f"{field}=None/empty")
            elif field == 'rank' and not isinstance(comp[field], (int, float)):
                invalid_fields.append(f"{field}=non-numeric")
            elif field in ['symbol', 'name'] and not isinstance(comp[field], str):
                invalid_fields.append(f"{field}=non-string")

        if missing_fields:
            errors.append(f"Component {i+1}: Missing fields: {missing_fields}")
        if invalid_fields:
            errors.append(f"Component {i+1}: Invalid fields: {invalid_fields}")

    if errors:
        return False, f"Validation failed: {'; '.join(errors)}"

    # Enhanced data extraction with null safety
    symbols = []
    ranks = []
    weights = []

    for comp in components:
        # Safe symbol extraction
        symbol = comp.get('symbol')
        if symbol and isinstance(symbol, str) and symbol.strip():
            symbols.append(symbol.strip().upper())

        # Safe rank extraction
        rank = comp.get('rank')
        if rank is not None and isinstance(rank, (int, float)) and not pd.isna(rank):
            ranks.append(int(rank))
        else:
            ranks.append(0)  # Default for missing ranks

        # Safe weight extraction
        weight = comp.get('weight')
        if weight is not None and isinstance(weight, (int, float)) and not pd.isna(weight) and weight > 0:
            weights.append(float(weight))

    print(f"📊 Extracted: {len(symbols)} symbols, {len(ranks)} ranks, {len(weights)} weights")

    # Check for duplicates
    duplicate_symbols = set([x for x in symbols if symbols.count(x) > 1])
    if duplicate_symbols:
        errors.append(f"Duplicate symbols found: {duplicate_symbols}")

    # Check rank consistency
    if len(set(ranks)) != len(ranks):
        duplicate_ranks = set([x for x in ranks if ranks.count(x) > 1])
        warnings.append(f"Duplicate ranks found: {duplicate_ranks}")

    expected_ranks = set(range(1, len(components) + 1))
    actual_ranks = set(ranks)
    missing_ranks = expected_ranks - actual_ranks
    extra_ranks = actual_ranks - expected_ranks

    if missing_ranks:
        warnings.append(f"Missing expected ranks: {sorted(missing_ranks)}")
    if extra_ranks:
        warnings.append(f"Unexpected ranks found: {sorted(extra_ranks)}")

    # Enhanced weight validation with null safety
    if weights:
        try:
            # Filter out any remaining invalid weights
            valid_weights = [w for w in weights if isinstance(w, (int, float)) and not pd.isna(w) and w > 0]

            if valid_weights:
                weight_sum = sum(valid_weights)
                if weight_sum < 80 or weight_sum > 120:
                    warnings.append(f"Total weight ({weight_sum:.2f}%) outside expected range (80-120%)")

                max_weight = max(valid_weights)
                min_weight = min(valid_weights)
                if max_weight > 15:  # QQQ top holdings typically under 15%
                    warnings.append(f"Unusually high individual weight: {max_weight:.2f}%")
                if min_weight <= 0:
                    warnings.append(f"Found zero or negative weights (min: {min_weight:.4f}%)")

                # Check for reasonable distribution
                if len(valid_weights) > 10:
                    avg_weight = weight_sum / len(valid_weights)
                    if avg_weight > 5:  # QQQ should have many small positions
                        warnings.append(f"Average weight ({avg_weight:.2f}%) unusually high for QQQ")
            else:
                warnings.append("No valid weight data found after filtering")
        except Exception as e:
            warnings.append(f"Error validating weights: {e}")
    else:
        warnings.append("No weight data available for validation")

    # Expected QQQ characteristics
    if len(components) < 90:
        warnings.append(f"QQQ typically has ~100 components, found only {len(components)}")
    elif len(components) > 110:
        warnings.append(f"QQQ typically has ~100 components, found {len(components)}")

    # Check for expected major components (should be in top 10)
    major_qqq_symbols = {'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'TSLA', 'GOOG', 'GOOGL'}
    top_10_symbols = set([comp['symbol'] for comp in components[:10]])
    missing_majors = major_qqq_symbols - top_10_symbols

    if missing_majors:
        warnings.append(f"Expected major QQQ components not in top 10: {missing_majors}")

    # Compile report
    report.append(f"✓ Validated {len(components)} QQQ components")
    report.append(f"✓ {len(symbols)} unique symbols")
    if weights:
        report.append(f"✓ Weight data available for {len(weights)} components (total: {sum(weights):.2f}%)")

    if warnings:
        report.append(f"⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            report.append(f"   - {warning}")

    if errors:
        report.append(f"❌ Errors ({len(errors)}):")
        for error in errors:
            report.append(f"   - {error}")
        return False, '\n'.join(report)

    is_valid = len(errors) == 0
    return is_valid, '\n'.join(report)

def clean_and_standardize_symbol(symbol):
    """
    Enhanced symbol cleaning with comprehensive null safety.
    """
    if symbol is None:
        return ""

    if not isinstance(symbol, str):
        try:
            symbol = str(symbol)
        except (ValueError, TypeError):
            print(f"⚠️  Warning: Could not convert symbol to string: {symbol} (type: {type(symbol)})")
            return ""

    if not symbol or not symbol.strip():
        return ""

    try:
        # Clean and standardize
        symbol = symbol.strip().upper()

        # Remove problematic characters but preserve valid ticker formats
        symbol = re.sub(r'\s+', '', symbol)  # Remove internal whitespace
        symbol = re.sub(r'[^A-Z0-9.-]', '', symbol)  # Keep only valid ticker characters

        # Additional validation
        if len(symbol) > 10 or len(symbol) < 1:
            print(f"⚠️  Warning: Unusual symbol length: '{symbol}' ({len(symbol)} chars)")

        return symbol
    except Exception as e:
        print(f"❌ Error cleaning symbol '{symbol}': {e}")
        return ""

# S&P 500 functionality removed - focusing on QQQ only

def download_qqq_holdings():
    """
    Enhanced QQQ holdings download with improved parsing and validation.
    Returns a list of holdings in standardized format.

    Features:
    - Multiple parsing strategies with fallbacks
    - Enhanced data validation and cleaning
    - Better error reporting and logging
    - Raw file preservation for debugging
    """
    # Invesco QQQ holdings URL
    url = "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ"

    print(f"🚀 Starting QQQ holdings download from: {url}")
    print(f"   Target: ~100 Nasdaq-100 components with weights")
    
    try:
        # Enhanced download with better error handling
        print(f"   Making request to Invesco QQQ data endpoint...")
        response = requests.get(url, timeout=30, headers={'User-Agent': 'QQQ-Monitor/1.0'})
        response.raise_for_status()

        if not response.content:
            raise ValueError("Received empty response from Invesco")

        print(f"   ✓ Downloaded {len(response.content)} bytes from Invesco")
        
        # Save the raw file for inspection
        today = datetime.datetime.now().strftime("%Y%m%d")
        raw_file = RAW_DIR / f"qqq_holdings_raw_{today}.xlsx"
        csv_file = RAW_DIR / f"qqq_holdings_raw_{today}.csv"
        
        with open(raw_file, 'wb') as f:
            f.write(response.content)
        print(f"Saved raw QQQ holdings to {raw_file}")
        
        # Also save raw content as text for inspection
        try:
            with open(csv_file, 'wb') as f:
                f.write(response.content)
            print(f"Saved raw QQQ content as {csv_file} for inspection")
        except:
            print("Could not save raw content as CSV")
        
        # Smart content detection and parsing strategy
        result = None
        parse_method = ""

        # Step 1: Detect content format
        content_preview = response.content[:1000].decode('utf-8', errors='ignore')
        is_csv_format = ',' in content_preview and ('Fund Ticker' in content_preview or 'Holding Ticker' in content_preview)
        is_excel_format = response.headers.get('content-type', '').startswith('application/vnd.openxmlformats')

        print(f"📋 Content analysis:")
        print(f"   Content-Type header: {response.headers.get('content-type', 'not specified')}")
        print(f"   Appears to be CSV: {is_csv_format}")
        print(f"   Appears to be Excel: {is_excel_format}")
        print(f"   Content preview: {content_preview[:200]}...")

        # Step 2: Try CSV parsing first (since Invesco QQQ data is actually CSV)
        if is_csv_format or not is_excel_format:
            try:
                print("🔄 Attempting CSV parsing (primary strategy)")
                # Optimize parsing by trying most likely scenarios first
                parsing_configs = [
                    ('utf-8', ',', 'Standard UTF-8 CSV'),
                    ('utf-8', '\t', 'UTF-8 Tab-separated'),
                    ('latin1', ',', 'Latin-1 CSV'),
                    ('cp1252', ',', 'Windows CSV')
                ]

                for encoding, delimiter, desc in parsing_configs:
                    try:
                        print(f"     Trying {desc} ({encoding}, '{delimiter}')")
                        df = pd.read_csv(io.BytesIO(response.content),
                                         encoding=encoding,
                                         delimiter=delimiter,
                                         on_bad_lines='skip')

                        # Enhanced validation - check if we got meaningful QQQ data
                        if (len(df.columns) > 5 and len(df) > 50 and
                            any('ticker' in str(col).lower() for col in df.columns)):
                            print(f"✓ Successfully parsed QQQ data with {desc}")
                            print(f"   Found {len(df)} rows and {len(df.columns)} columns")
                            print(f"   Key columns: {df.columns.tolist()}")
                            # This success case is now handled above
                            
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
                            
                            if ticker_col:
                                print(f"Found key columns: Ticker={ticker_col}, Name={name_col}, Weight={weight_col}")
                                
                                # Basic cleanup
                                df = df.dropna(subset=[ticker_col])
                                
                                # Collect weight values for analysis with safe conversion
                                weight_vals = []
                                component_names = []
                                if weight_col:
                                    # Safe numeric conversion for weights
                                    weight_series = df[weight_col].dropna()
                                    weight_vals = pd.to_numeric(weight_series, errors='coerce').dropna().tolist()

                                    # Get component names for valid weights only
                                    valid_weight_mask = pd.to_numeric(df[weight_col], errors='coerce').notna()
                                    valid_df = df[valid_weight_mask & df[ticker_col].notna()]
                                    component_names = [f"{row[ticker_col]}: {row[name_col]}" if name_col and pd.notna(row[name_col]) else str(row[ticker_col])
                                                       for _, row in valid_df.iterrows()]
                                
                                # Normalize weights if there are any
                                normalized_weights = {}
                                if weight_vals:
                                    normalized = normalize_weights(weight_vals, component_names)
                                    # Create a dictionary mapping row index to normalized weight
                                    # Use the same valid weight mask to ensure consistency
                                    valid_weight_mask = pd.to_numeric(df[weight_col], errors='coerce').notna()
                                    weight_indices = df[valid_weight_mask].index
                                    normalized_weights = {idx: normalized[i] for i, idx in enumerate(weight_indices) if i < len(normalized)}
                                
                                # Format data into standardized structure
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
                                        
                                        # Add weight if available - use normalized value with safety checks
                                        if weight_col and pd.notna(row.get(weight_col)) and i in normalized_weights:
                                            weight_value = normalized_weights[i]
                                            if isinstance(weight_value, (int, float)) and not (pd.isna(weight_value) or np.isinf(weight_value)):
                                                item["weight"] = weight_value
                                            else:
                                                print(f"   ⚠️  Skipping invalid weight for {symbol}: {weight_value}")
                                        
                                        result.append(item)
                                    except Exception as e:
                                        print(f"Error processing row {i} with CSV approach: {e}")
                                        continue
                                
                                print(f"✓ Processed {len(result)} QQQ holdings successfully with {desc}")
                                parse_method = f"csv-{encoding}-{delimiter}"
                                # Early exit - we found working data, no need to try other configs
                                print(f"✅ CSV parsing successful, skipping remaining attempts")
                                break
                    except Exception as e:
                        print(f"   ❌ {desc} failed: {str(e)[:100]}...")

                    # Break outer loop if we found a working approach
                    if result:
                        print(f"🎉 Successfully parsed QQQ data using CSV strategy")
                        break
            except Exception as e:
                print(f"⚠️  CSV parsing approach failed: {e}")

        # Step 3: Fallback to Excel parsing if CSV failed
        if not result:
            print("🔄 Falling back to Excel parsing strategies")
            try:
                print("   Attempting standard Excel parsing (skiprows=1)")
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

                    # Collect weight values for analysis with safe conversion
                    weight_vals = []
                    component_names = []
                    if weight_col:
                        # Safe numeric conversion for weights
                        weight_series = df[weight_col].dropna()
                        weight_vals = pd.to_numeric(weight_series, errors='coerce').dropna().tolist()

                        # Get component names for valid weights only
                        valid_weight_mask = pd.to_numeric(df[weight_col], errors='coerce').notna()
                        valid_df = df[valid_weight_mask & df[ticker_col].notna()]
                        component_names = [f"{row[ticker_col]}: {row[name_col]}" if name_col and pd.notna(row[name_col]) else str(row[ticker_col])
                                           for _, row in valid_df.iterrows()]
                    
                    # Normalize weights if there are any
                    normalized_weights = {}
                    if weight_vals:
                        normalized = normalize_weights(weight_vals, component_names)
                        # Create a dictionary mapping row index to normalized weight
                        # Use the same valid weight mask to ensure consistency
                        valid_weight_mask = pd.to_numeric(df[weight_col], errors='coerce').notna()
                        weight_indices = df[valid_weight_mask].index
                        normalized_weights = {idx: normalized[i] for i, idx in enumerate(weight_indices) if i < len(normalized)}

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
                            
                            # Add weight if available - use normalized value with safety checks
                            if weight_col and pd.notna(row.get(weight_col)) and i in normalized_weights:
                                weight_value = normalized_weights[i]
                                if isinstance(weight_value, (int, float)) and not (pd.isna(weight_value) or np.isinf(weight_value)):
                                    item["weight"] = weight_value
                                else:
                                    print(f"   ⚠️  Skipping invalid weight for {symbol}: {weight_value}")

                            result.append(item)
                        except Exception as e:
                            print(f"Error processing QQQ row {i}: {e}")
                            continue
                    
                    print(f"Processed {len(result)} QQQ holdings with standard approach")
                    parse_method = "standard"
                else:
                    print("   ⚠️  Standard Excel parsing failed - missing expected columns")
                    print(f"   Found columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"   ❌ Standard Excel parsing failed: {e}")

        # Step 4: Try alternative Excel parsing with different skiprows
        if not result:
            print("   Attempting alternative Excel parsing strategies")
            for skiprows in [0, 2, 3, 4]:
                try:
                    print(f"   Trying skiprows={skiprows}")
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
                        # Basic cleanup
                        df = df.dropna(subset=[ticker_col])
                        
                        # Collect weight values for analysis with safe conversion
                        weight_vals = []
                        component_names = []
                        if weight_col:
                            # Safe numeric conversion for weights
                            weight_series = df[weight_col].dropna()
                            weight_vals = pd.to_numeric(weight_series, errors='coerce').dropna().tolist()

                            # Get component names for valid weights only
                            valid_weight_mask = pd.to_numeric(df[weight_col], errors='coerce').notna()
                            valid_df = df[valid_weight_mask & df[ticker_col].notna()]
                            component_names = [f"{row[ticker_col]}: {row[name_col]}" if name_col and pd.notna(row[name_col]) else str(row[ticker_col])
                                               for _, row in valid_df.iterrows()]
                        
                        # Normalize weights if there are any
                        normalized_weights = {}
                        if weight_vals:
                            normalized = normalize_weights(weight_vals, component_names)
                            # Create a dictionary mapping row index to normalized weight
                            # Use the same valid weight mask to ensure consistency
                            valid_weight_mask = pd.to_numeric(df[weight_col], errors='coerce').notna()
                            weight_indices = df[valid_weight_mask].index
                            normalized_weights = {idx: normalized[i] for i, idx in enumerate(weight_indices) if i < len(normalized)}
                        
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
                                
                                # Add weight if available - use normalized value
                                if weight_col and pd.notna(row[weight_col]) and i in normalized_weights:
                                    item["weight"] = normalized_weights[i]
                                
                                result.append(item)
                            except Exception as e:
                                print(f"Error processing row {i} with alternative approach: {e}")
                                continue
                        
                        print(f"Processed {len(result)} QQQ holdings with alternative approach (skiprows={skiprows})")
                        parse_method = f"alternative-{skiprows}"
                        break
                except Exception as e:
                    print(f"   ❌ skiprows={skiprows} failed: {e}")

        # Step 5: Try openpyxl engine as final Excel fallback
        if not result:
            try:
                print("   Attempting openpyxl engine parsing")
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
                    
                    # Collect weight values for analysis with safe conversion
                    weight_vals = []
                    component_names = []
                    if weight_col:
                        # Safe numeric conversion for weights
                        weight_series = df[weight_col].dropna()
                        weight_vals = pd.to_numeric(weight_series, errors='coerce').dropna().tolist()

                        # Get component names for valid weights only
                        valid_weight_mask = pd.to_numeric(df[weight_col], errors='coerce').notna()
                        valid_df = df[valid_weight_mask & df[ticker_col].notna()]
                        component_names = [f"{row[ticker_col]}: {row[name_col]}" if name_col and pd.notna(row[name_col]) else str(row[ticker_col])
                                           for _, row in valid_df.iterrows()]

                    # Normalize weights if there are any
                    normalized_weights = {}
                    if weight_vals:
                        normalized = normalize_weights(weight_vals, component_names)
                        # Create a dictionary mapping row index to normalized weight
                        # Use the same valid weight mask to ensure consistency
                        valid_weight_mask = pd.to_numeric(df[weight_col], errors='coerce').notna()
                        weight_indices = df[valid_weight_mask].index
                        normalized_weights = {idx: normalized[i] for i, idx in enumerate(weight_indices) if i < len(normalized)}
                    
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
                            
                            # Add weight if available - use normalized value with safety checks
                            if weight_col and pd.notna(row.get(weight_col)) and i in normalized_weights:
                                weight_value = normalized_weights[i]
                                if isinstance(weight_value, (int, float)) and not (pd.isna(weight_value) or np.isinf(weight_value)):
                                    item["weight"] = weight_value
                                else:
                                    print(f"   ⚠️  Skipping invalid weight for {symbol}: {weight_value}")

                            result.append(item)
                        except Exception as e:
                            print(f"Error processing row {i} with openpyxl engine: {e}")
                            continue
                    
                    print(f"Processed {len(result)} QQQ holdings with openpyxl engine")
                    parse_method = "openpyxl"
            except Exception as e:
                print(f"   ❌ openpyxl engine failed: {e}")
        
        # Enhanced result processing and validation
        if result and len(result) > 0:
            print(f"✓ Initial parsing successful: {len(result)} components found")

            # Clean and standardize symbols
            for item in result:
                if 'symbol' in item:
                    original_symbol = item['symbol']
                    item['symbol'] = clean_and_standardize_symbol(item['symbol'])
                    if item['symbol'] != original_symbol:
                        print(f"   Cleaned symbol: {original_symbol} → {item['symbol']}")

            # Remove any components without valid symbols
            result = [item for item in result if item.get('symbol', '').strip()]
            print(f"✓ After symbol validation: {len(result)} components")

            # Sort by rank and ensure proper ranking
            result.sort(key=lambda x: x.get("rank", 999))

            # Re-assign ranks to ensure consistency (1-based)
            for i, item in enumerate(result):
                item["rank"] = i + 1

            # Validate the data quality
            is_valid, validation_report = validate_qqq_data(result)
            print("\n" + "─" * 50)
            print("QQQ DATA VALIDATION REPORT")
            print("─" * 50)
            print(validation_report)
            print("─" * 50)

            if not is_valid:
                print("⚠️  Data validation found critical issues, but proceeding with available data")

            # Save parsed data with validation report
            parsed_file = RAW_DIR / f"qqq_holdings_parsed_{today}_{parse_method}.json"
            parsed_data = {
                "metadata": {
                    "download_date": today,
                    "parse_method": parse_method,
                    "total_components": len(result),
                    "validation_passed": is_valid,
                    "source_url": url
                },
                "validation_report": validation_report.split('\n'),
                "holdings": result
            }

            with open(parsed_file, 'w') as f:
                json.dump(parsed_data, f, indent=2)
            print(f"✓ Saved parsed QQQ holdings with validation report to {parsed_file}")

            print(f"✓ Successfully processed {len(result)} QQQ holdings using method: {parse_method}")
            return result
        else:
            print("❌ All parsing attempts failed for QQQ holdings")
            print("   Examining common failure modes...")

            # Enhanced debugging information
            try:
                if response:
                    print(f"   Response status: {response.status_code}")
                    print(f"   Response headers: {dict(response.headers)}")

                    content_snippet = response.content[:500] if response.content else b"No content"
                    print(f"   Response content preview: {content_snippet}")

                    # Enhanced content analysis
                    content_str = content_snippet.decode('utf-8', errors='ignore').lower()
                    if "<html" in content_str or "<!doctype" in content_str:
                        print("   ⚠️  Response appears to be HTML - possible redirect or error page")
                    elif "access denied" in content_str or "forbidden" in content_str:
                        print("   ⚠️  Response indicates access denied - possible IP blocking")
                    elif "rate limit" in content_str or "too many requests" in content_str:
                        print("   ⚠️  Response indicates rate limiting")
                    elif not any(indicator in content_str for indicator in ["fund", "ticker", "weight", "holding"]):
                        print("   ⚠️  Response doesn't contain expected financial data indicators")
                else:
                    print("   ❌ No response object available")
            except Exception as debug_e:
                print(f"   ❌ Error analyzing response: {debug_e}")

            print("   Falling back to hardcoded QQQ data...")
            return get_qqq_components_hardcoded()
    
    except Exception as e:
        print(f"Error downloading QQQ holdings: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to hardcoded data if download fails
        print("Falling back to hardcoded QQQ data...")
        return get_qqq_components_hardcoded()

# S&P 500 hardcoded data removed - focusing on QQQ only

def get_qqq_components_hardcoded():
    """
    Updated hardcoded QQQ components as fallback with more recent data.
    """
    print("Using hardcoded QQQ top components...")

    # Updated top QQQ components with more recent approximate weights (as of 2024)
    qqq_components = [
        {"symbol": "AAPL", "name": "Apple Inc.", "weight": 11.94, "rank": 1},
        {"symbol": "MSFT", "name": "Microsoft Corporation", "weight": 10.20, "rank": 2},
        {"symbol": "NVDA", "name": "NVIDIA Corporation", "weight": 7.52, "rank": 3},
        {"symbol": "AMZN", "name": "Amazon.com Inc.", "weight": 6.82, "rank": 4},
        {"symbol": "META", "name": "Meta Platforms Inc. Class A", "weight": 4.99, "rank": 5},
        {"symbol": "TSLA", "name": "Tesla Inc.", "weight": 3.53, "rank": 6},
        {"symbol": "GOOG", "name": "Alphabet Inc. Class C", "weight": 3.49, "rank": 7},
        {"symbol": "GOOGL", "name": "Alphabet Inc. Class A", "weight": 3.15, "rank": 8},
        {"symbol": "AVGO", "name": "Broadcom Inc.", "weight": 2.79, "rank": 9},
        {"symbol": "COST", "name": "Costco Wholesale Corporation", "weight": 2.21, "rank": 10},
        {"symbol": "ASML", "name": "ASML Holding N.V.", "weight": 1.95, "rank": 11},
        {"symbol": "ADBE", "name": "Adobe Inc.", "weight": 1.92, "rank": 12},
        {"symbol": "CSCO", "name": "Cisco Systems Inc.", "weight": 1.87, "rank": 13},
        {"symbol": "AMD", "name": "Advanced Micro Devices Inc.", "weight": 1.49, "rank": 14},
        {"symbol": "PEP", "name": "PepsiCo Inc.", "weight": 1.47, "rank": 15},
        {"symbol": "NFLX", "name": "Netflix Inc.", "weight": 1.38, "rank": 16},
        {"symbol": "TMUS", "name": "T-Mobile US Inc.", "weight": 1.37, "rank": 17},
        {"symbol": "CMCSA", "name": "Comcast Corporation Class A", "weight": 1.33, "rank": 18},
        {"symbol": "INTU", "name": "Intuit Inc.", "weight": 1.19, "rank": 19},
        {"symbol": "QCOM", "name": "QUALCOMM Incorporated", "weight": 1.15, "rank": 20}
    ]
    return qqq_components

# S&P 500 component function removed - focusing on QQQ only

def get_qqq_components():
    """
    Enhanced QQQ component retrieval with improved data quality and error handling.

    Features:
    - Official ETF data download with multiple fallback strategies
    - Enhanced market cap integration with better rate limiting
    - Comprehensive data validation and quality checks
    - Detailed logging and progress reporting
    """
    print("📈 Getting QQQ (Nasdaq-100) components...")

    try:
        # Step 1: Download from official ETF provider
        print("Step 1: Downloading from official Invesco source...")
        components = download_qqq_holdings()

        if not components:
            print("⚠️  Official download failed, using hardcoded fallback...")
            components = get_qqq_components_hardcoded()
        else:
            print(f"✓ Successfully retrieved {len(components)} components from official source")

        # Step 2: Market cap enhancement for top holdings
        print("\nStep 2: Enhancing with real-time market cap data...")
        top_symbols = [component["symbol"] for component in components[:15]]  # Increased to top 15
        print(f"   Fetching market cap data for top {len(top_symbols)} holdings...")

        # Fetch real data with improved error handling
        real_data = fetch_top_stocks_data(top_symbols, max_stocks=15)

        if real_data:
            print(f"✓ Retrieved market cap data for {len(real_data)} symbols")

            # Update components with real data while preserving ETF-specific data
            enhanced_count = 0
            for i, component in enumerate(components):
                symbol = component["symbol"]
                if symbol in real_data:
                    # Preserve original ETF data while adding market cap info
                    market_data = real_data[symbol]
                    component["market_cap"] = market_data.get("market_cap", 0)
                    # Use real company name if available and more accurate
                    if market_data.get("name") and len(market_data["name"]) > len(component.get("name", "")):
                        component["name"] = market_data["name"]
                    component["real_data"] = True
                    enhanced_count += 1

            print(f"✓ Enhanced {enhanced_count} components with real-time market data")
        else:
            print("⚠️  No market cap data retrieved (API limits or errors)")

        # Step 3: Final validation and quality check
        print("\nStep 3: Final data quality validation...")
        is_valid, validation_report = validate_qqq_data(components)

        if is_valid:
            print("✓ Final validation passed")
        else:
            print("⚠️  Final validation found issues - check logs")

        # Summary
        total_with_weights = sum(1 for c in components if 'weight' in c and c.get('weight', 0) > 0)
        total_with_market_cap = sum(1 for c in components if 'market_cap' in c and c.get('market_cap', 0) > 0)

        print(f"\n📋 QQQ DATA SUMMARY:")
        print(f"   Total components: {len(components)}")
        print(f"   With weight data: {total_with_weights}")
        print(f"   With market cap data: {total_with_market_cap}")
        print(f"   Data quality: {'Excellent' if is_valid else 'Acceptable with warnings'}")

        return components

    except Exception as e:
        print(f"❌ Critical error getting QQQ components: {e}")
        import traceback
        traceback.print_exc()
        print("   Using hardcoded fallback data...")
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
    
    # Check if there's an excessive number of changes (might indicate a problem)
    total_potential_changes = len(added) + len(removed)
    
    if total_potential_changes > MAX_REASONABLE_CHANGES:
        print(f"WARNING: Detected {total_potential_changes} potential additions/removals, which exceeds the threshold of {MAX_REASONABLE_CHANGES}.")
        print("This may indicate a data format change rather than actual index changes.")
        print("Only reporting changes in the top positions to avoid excessive notifications.")
        
        # In this case, limit additions/removals to top positions
        added = {symbol for symbol in added if curr_by_symbol[symbol]['rank'] <= TOP_POSITIONS_TO_TRACK * 2}
        removed = {symbol for symbol in removed if prev_by_symbol[symbol]['rank'] <= TOP_POSITIONS_TO_TRACK * 2}
    
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
        
        # Enhanced weight comparison with null safety
        prev_weight_raw = prev_by_symbol[symbol].get("weight")
        curr_weight_raw = curr_by_symbol[symbol].get("weight")

        # Safe weight change detection
        if (prev_weight_raw is not None and curr_weight_raw is not None and
            isinstance(prev_weight_raw, (int, float)) and isinstance(curr_weight_raw, (int, float)) and
            not pd.isna(prev_weight_raw) and not pd.isna(curr_weight_raw)):

            prev_weight = float(prev_weight_raw)
            curr_weight = float(curr_weight_raw)
            weight_change = curr_weight - prev_weight
            
            # Add a sanity check for unrealistic weight changes
            if abs(weight_change) > MAX_WEIGHT_THRESHOLD:
                print(f"WARNING: Unrealistic weight change for {symbol}: {prev_weight:.2f}% -> {curr_weight:.2f}% ({weight_change:+.2f}%)")
                print("This may indicate a data format issue. Ignoring this weight change.")
            elif abs(weight_change) > 0.1:  # More than 0.1 percentage point
                name = curr_by_symbol[symbol]["name"]
                weight_msg = f"Weight changed from {prev_weight:.2f}% to {curr_weight:.2f}% ({weight_change:+.2f}%)"
                weight_change_msg = f"WEIGHT CHANGE: {name} ({symbol}) - {weight_msg}"
                changes["message"].append(weight_change_msg)
                changes["weight_changes"].append(weight_change_msg)
                
                # Special focus on significant weight changes in top positions
                if prev_rank <= TOP_POSITIONS_TO_TRACK or curr_rank <= TOP_POSITIONS_TO_TRACK:
                    if abs(weight_change) > 0.5:  # More significant threshold for top positions
                        changes["top_changes"].append(f"TOP {TOP_POSITIONS_TO_TRACK} SIGNIFICANT WEIGHT CHANGE: {name} ({symbol}) {weight_msg}")
        
        # Enhanced market cap change detection with null safety
        prev_mcap_raw = prev_by_symbol[symbol].get("market_cap")
        curr_mcap_raw = curr_by_symbol[symbol].get("market_cap")

        if (prev_mcap_raw is not None and curr_mcap_raw is not None and
            isinstance(prev_mcap_raw, (int, float)) and isinstance(curr_mcap_raw, (int, float)) and
            not pd.isna(prev_mcap_raw) and not pd.isna(curr_mcap_raw) and
            prev_mcap_raw > 0 and curr_mcap_raw > 0):

            prev_mcap = float(prev_mcap_raw)
            curr_mcap = float(curr_mcap_raw)

            # Calculate percentage change with safety check
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

def format_top_holdings(current_data, top_n=10):
    """
    Format top N holdings into an HTML table for email display.

    Args:
        current_data: List of component dictionaries with rank, symbol, name, weight, market_cap
        top_n: Number of top holdings to display (default: 10)

    Returns:
        HTML string with formatted leaderboard table
    """
    if not current_data or not isinstance(current_data, list):
        return '<p style="color: #6c757d; font-style: italic;">⚠️ Top holdings data unavailable</p>'

    try:
        # Sort by rank and get top N
        sorted_data = sorted(current_data, key=lambda x: x.get('rank', 999))
        top_holdings = sorted_data[:top_n]

        if not top_holdings:
            return '<p style="color: #6c757d; font-style: italic;">⚠️ No holdings data to display</p>'

        # Trophy emojis for top 3
        rank_badges = {1: '🥇', 2: '🥈', 3: '🥉'}

        # Build HTML table rows
        rows_html = []
        for holding in top_holdings:
            rank = holding.get('rank', 0)
            symbol = holding.get('symbol', 'N/A')
            name = holding.get('name', 'Unknown')
            weight = holding.get('weight')
            market_cap = holding.get('market_cap')

            # Format rank with badge
            rank_display = f"{rank_badges.get(rank, '')} {rank}" if rank else 'N/A'

            # Format weight with visual bar
            if weight and isinstance(weight, (int, float)) and not pd.isna(weight):
                weight_pct = float(weight)
                weight_display = f"{weight_pct:.2f}%"

                # Calculate bar width (max 100px for 15% weight)
                bar_width = min(int((weight_pct / 15) * 100), 100)

                # Color code based on weight
                if weight_pct >= 5:
                    bar_color = '#28a745'  # Green for large holdings
                elif weight_pct >= 2:
                    bar_color = '#007bff'  # Blue for medium holdings
                else:
                    bar_color = '#6c757d'  # Gray for small holdings

                weight_bar = f'<div class="weight-bar-container"><div class="weight-bar" style="width: {bar_width}px; background: {bar_color};"></div></div>'
            else:
                weight_display = 'N/A'
                weight_bar = ''

            # Format market cap
            if market_cap and isinstance(market_cap, (int, float)) and not pd.isna(market_cap) and market_cap > 0:
                mcap_billions = float(market_cap) / 1e9
                if mcap_billions >= 1000:
                    mcap_display = f"${mcap_billions/1000:.2f}T"
                else:
                    mcap_display = f"${mcap_billions:.1f}B"
            else:
                mcap_display = '<span style="color: #adb5bd;">—</span>'

            # Truncate long company names
            if len(name) > 30:
                name = name[:27] + '...'

            # Build row HTML
            row_html = f"""
                <tr>
                    <td class="rank-cell">{rank_display}</td>
                    <td class="symbol-cell">{symbol}</td>
                    <td class="name-cell">{name}</td>
                    <td class="weight-cell">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="min-width: 50px; text-align: right; font-weight: 600;">{weight_display}</span>
                            {weight_bar}
                        </div>
                    </td>
                    <td class="mcap-cell">{mcap_display}</td>
                </tr>
            """
            rows_html.append(row_html)

        # Combine into table
        table_html = f"""
        <table class="holdings-table">
            <thead>
                <tr>
                    <th style="text-align: center;">Rank</th>
                    <th>Symbol</th>
                    <th>Company</th>
                    <th>Weight</th>
                    <th>Market Cap</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
        """

        return table_html

    except Exception as e:
        print(f"❌ Error formatting top holdings: {e}")
        import traceback
        traceback.print_exc()
        return f'<p style="color: #dc3545;">⚠️ Error displaying holdings: {str(e)}</p>'

def send_email_alert(index_name, changes, current_data=None):
    """Enhanced email alert with top holdings leaderboard and improved categorization."""
    if not changes or not changes["message"] or changes["message"][0] == "Initial data collection - no changes to report.":
        print(f"📫 No changes to report for {index_name}")
        return

    # Check email configuration
    missing_config = []
    if not EMAIL_SENDER:
        missing_config.append("EMAIL_SENDER")
    if not EMAIL_PASSWORD:
        missing_config.append("EMAIL_PASSWORD or EMAIL_APP_PASSWORD")
    if not EMAIL_RECIPIENT:
        missing_config.append("EMAIL_RECIPIENT")

    if missing_config:
        print(f"❌ Email configuration missing: {', '.join(missing_config)}")
        print("   Please set the required environment variables to enable email alerts")
        return

    print(f"📧 Preparing enhanced email alert for {index_name} changes...")
    print(f"   Sender: {EMAIL_SENDER}")
    print(f"   Recipient: {EMAIL_RECIPIENT}")
    print(f"   Total changes: {len(changes['message'])}")
    print(f"   Top position changes: {len(changes['top_changes'])}")
    print(f"   Including top holdings leaderboard: {bool(current_data)}")
    if current_data:
        print(f"   Current data available: {len(current_data)} components")

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.datetime.now().strftime("%H:%M UTC")

    # Enhanced subject line with priority and change count
    priority_indicator = "🔥" if changes["top_changes"] else "📈"
    change_summary = f"{len(changes['top_changes'])} top" if changes["top_changes"] else f"{len(changes['message'])} total"

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = f"{priority_indicator} {index_name} Alert: {change_summary} changes - {today}"
    
    # Enhanced HTML email body with modern styling
    body = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
            }}
            .header h1 {{ margin: 0; font-size: 24px; }}
            .header .subtitle {{ margin: 5px 0 0 0; opacity: 0.9; font-size: 14px; }}
            .section {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                border-left: 4px solid #007bff;
            }}
            .critical-section {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
            }}
            .top-change {{
                background: linear-gradient(90deg, #ff6b6b, #ffa726);
                color: white;
                padding: 12px;
                margin: 8px 0;
                border-radius: 6px;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .regular-change {{
                padding: 8px 12px;
                margin: 4px 0;
                border-radius: 4px;
                background: white;
                border-left: 3px solid #17a2b8;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-box {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-number {{
                font-size: 24px;
                font-weight: bold;
                color: #007bff;
                margin-bottom: 5px;
            }}
            .stat-label {{
                font-size: 12px;
                text-transform: uppercase;
                color: #6c757d;
                margin: 0;
            }}
            h2 {{ color: #1e3c72; margin-bottom: 10px; }}
            h3 {{ color: #2a5298; margin: 20px 0 10px 0; }}
            .emoji {{ font-size: 18px; margin-right: 8px; }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding: 15px;
                background: #e9ecef;
                border-radius: 8px;
                font-size: 12px;
                color: #6c757d;
            }}
            /* Top Holdings Table Styles */
            .holdings-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .holdings-table thead {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
            }}
            .holdings-table th {{
                padding: 12px;
                text-align: left;
                font-weight: 600;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .holdings-table tbody tr {{
                border-bottom: 1px solid #e9ecef;
                transition: background-color 0.2s;
            }}
            .holdings-table tbody tr:hover {{
                background-color: #f8f9fa;
            }}
            .holdings-table tbody tr:nth-child(even) {{
                background-color: #fafbfc;
            }}
            .holdings-table td {{
                padding: 12px;
                font-size: 14px;
            }}
            .rank-cell {{
                font-weight: bold;
                font-size: 16px;
                text-align: center;
                width: 80px;
            }}
            .symbol-cell {{
                font-weight: bold;
                color: #007bff;
                font-family: 'Courier New', monospace;
                width: 80px;
            }}
            .name-cell {{
                color: #495057;
                width: 250px;
            }}
            .weight-cell {{
                width: 200px;
            }}
            .mcap-cell {{
                text-align: right;
                font-weight: 600;
                color: #28a745;
                width: 100px;
            }}
            .weight-bar-container {{
                flex-grow: 1;
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
                overflow: hidden;
            }}
            .weight-bar {{
                height: 100%;
                border-radius: 4px;
                transition: width 0.3s ease;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📈 {index_name} Change Alert</h1>
            <div class="subtitle">Changes detected on {today} at {time_now}</div>
        </div>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-number">{len(changes.get('top_changes', []))}</div>
                <div class="stat-label">Top {TOP_POSITIONS_TO_TRACK} Changes</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len(changes.get('message', []))}</div>
                <div class="stat-label">Total Changes</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len(changes.get('additions', [])) + len(changes.get('removals', []))}</div>
                <div class="stat-label">Additions + Removals</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{len(changes.get('rank_changes', []))}</div>
                <div class="stat-label">Rank Changes</div>
            </div>
        </div>

        <!-- Top 10 Holdings Leaderboard -->
        <div class="section" style="background: linear-gradient(to right, #f8f9fa, #ffffff); border-left: 4px solid #667eea;">
            <h3><span class="emoji">🏆</span>Current Top 10 Holdings</h3>
            <p style="margin-bottom: 15px; color: #6c757d;">Live snapshot of the most valuable components in the {index_name} index</p>
            {format_top_holdings(current_data, top_n=10)}
        </div>
    """

    # Add top position changes with enhanced highlighting
    if changes["top_changes"]:
        body += f'<div class="section critical-section">'
        body += f'<h3><span class="emoji">🔥</span>Critical: Top {TOP_POSITIONS_TO_TRACK} Position Changes</h3>\n'
        body += f'<p><strong>These changes affect the most influential components of the index.</strong></p>\n'
        for change in changes["top_changes"]:
            body += f'<div class="top-change">{change}</div>\n'
        body += '</div>'
    
    # Enhanced categorized sections
    # Rank changes
    rank_changes = [c for c in changes["rank_changes"] if c not in [tc for tc in changes["top_changes"] if "MOVEMENT" in tc]]
    if rank_changes:
        body += '<div class="section">'
        body += f'<h3><span class="emoji">📊</span>Position Changes ({len(rank_changes)})</h3>\n'
        body += f'<p>Companies that moved up or down in the index ranking.</p>\n'
        for change in rank_changes:
            body += f'<div class="regular-change">{change}</div>\n'
        body += '</div>'
    
    # Weight changes
    weight_changes = [c for c in changes["weight_changes"] if c not in [tc for tc in changes["top_changes"] if "WEIGHT" in tc]]
    if weight_changes:
        body += '<div class="section">'
        body += f'<h3><span class="emoji">⚖️</span>Weight Adjustments ({len(weight_changes)})</h3>\n'
        body += f'<p>Significant changes in component weightings within the index.</p>\n'
        for change in weight_changes:
            body += f'<div class="regular-change">{change}</div>\n'
        body += '</div>'
    
    # Market cap changes
    market_cap_changes = [c for c in changes["market_cap_changes"] if c not in [tc for tc in changes["top_changes"] if "MARKET CAP" in tc]]
    if market_cap_changes:
        body += '<div class="section">'
        body += f'<h3><span class="emoji">💰</span>Market Cap Movements ({len(market_cap_changes)})</h3>\n'
        body += f'<p>Notable changes in company valuations affecting their index presence.</p>\n'
        for change in market_cap_changes:
            body += f'<div class="regular-change">{change}</div>\n'
        body += '</div>'
    
    # Additions
    additions = [c for c in changes["additions"] if c not in [tc for tc in changes["top_changes"] if "ADDITION" in tc]]
    if additions:
        body += '<div class="section">'
        body += f'<h3><span class="emoji">➕</span>New Components ({len(additions)})</h3>\n'
        body += f'<p>Companies added to the {index_name} index.</p>\n'
        for change in additions:
            body += f'<div class="regular-change">{change}</div>\n'
        body += '</div>'
    
    # Removals
    removals = [c for c in changes["removals"] if c not in [tc for tc in changes["top_changes"] if "REMOVAL" in tc]]
    if removals:
        body += '<div class="section">'
        body += f'<h3><span class="emoji">➖</span>Removed Components ({len(removals)})</h3>\n'
        body += f'<p>Companies no longer part of the {index_name} index.</p>\n'
        for change in removals:
            body += f'<div class="regular-change">{change}</div>\n'
        body += '</div>'
    
    # Other uncategorized changes
    other_changes = [c for c in changes["message"]
                    if c not in changes["top_changes"]
                    and c not in rank_changes
                    and c not in weight_changes
                    and c not in market_cap_changes
                    and c not in changes["additions"]
                    and c not in changes["removals"]]
    if other_changes:
        body += '<div class="section">'
        body += f'<h3><span class="emoji">ℹ️</span>Other Changes ({len(other_changes)})</h3>\n'
        body += f'<p>Additional changes that don\'t fit the main categories above.</p>\n'
        for change in other_changes:
            body += f'<div class="regular-change">{change}</div>\n'
        body += '</div>'
    
    # Enhanced footer with more information
    data_source = "Invesco QQQ ETF" if "QQQ" in index_name else "Official ETF Provider"
    body += f"""
        <div class="footer">
            <h4>🤖 Automated Alert System</h4>
            <p><strong>Data Source:</strong> {data_source} Official Holdings Data</p>
            <p><strong>Generated:</strong> {today} at {time_now}</p>
            <p><strong>Focus:</strong> Tracking changes in top {TOP_POSITIONS_TO_TRACK} positions with enhanced accuracy</p>
            <p><strong>Technology:</strong> Enhanced parsing with multiple validation layers</p>
            <hr style="margin: 15px 0; border: none; border-top: 1px solid #dee2e6;">
            <p style="font-size: 10px;">This alert was generated automatically using official ETF provider data.<br>
            The system monitors composition changes, weight adjustments, and market cap movements.<br>
            For questions or to modify alert preferences, check your repository settings.</p>
        </div>
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
        print(f"✓ Enhanced email alert sent successfully for {index_name} changes")
        print(f"   Subject: {msg['Subject']}")
        print(f"   Content: {len(body)} characters with modern HTML formatting")
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
        
        # If this is the first run or the data format appears to have changed dramatically
        if not previous_data or (previous_data and current_data and 
                                 abs(len(previous_data) - len(current_data)) > MAX_REASONABLE_CHANGES):
            print(f"Initial data collection or major format change for {index_name} - saving for future comparison")
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
            
            # Determine if changes seem suspicious (too many)
            too_many_changes = len(changes["message"]) > MAX_REASONABLE_CHANGES
            if too_many_changes:
                print(f"WARNING: Detected {len(changes['message'])} changes, which exceeds the threshold of {MAX_REASONABLE_CHANGES}.")
                print("This may indicate a data format change rather than actual index changes.")
                print("Only changes in top positions will be reported.")
                
                # Reset the changes to only include top position changes
                filtered_changes = {
                    "message": changes["top_changes"].copy(),
                    "top_changes": changes["top_changes"].copy(),
                    "additions": [c for c in changes["additions"] if any(tc in c for tc in changes["top_changes"])],
                    "removals": [c for c in changes["removals"] if any(tc in c for tc in changes["top_changes"])],
                    "rank_changes": [c for c in changes["rank_changes"] if any(tc in c for tc in changes["top_changes"])],
                    "weight_changes": [c for c in changes["weight_changes"] if any(tc in c for tc in changes["top_changes"])],
                    "market_cap_changes": [c for c in changes["market_cap_changes"] if any(tc in c for tc in changes["top_changes"])]
                }
                changes = filtered_changes
            else:
                for change in changes["message"]:
                    if change not in changes["top_changes"]:
                        print(f"  - {change}")
        else:
            print(f"No {index_name} changes detected or initial data collection")
        
        # Save current data for next time
        print(f"Saving current data for {index_name}...")
        save_current_data(current_data, data_file)
        print(f"Data saved to {data_file}")

        # Return both changes and current_data for email display
        return changes, current_data

    except Exception as e:
        print(f"Error checking for changes in {index_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"message": [], "top_changes": []}, []

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
        
        # Check QQQ only (S&P 500 removed for simplification)
        qqq_changes, qqq_current_data = check_for_changes("QQQ (Nasdaq-100)", QQQ_FILE, get_qqq_components)
        send_email_alert("QQQ (Nasdaq-100)", qqq_changes, qqq_current_data)
        
        print("Index change check completed successfully")
        
    except Exception as e:
        print(f"Critical error in main function: {e}")

if __name__ == "__main__":
    main()