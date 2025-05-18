import pandas as pd
import requests
import io
import json
import os
import re
from pathlib import Path
from datetime import datetime

def download_and_analyze_etf_holdings(etf_type, save_formats=["xlsx", "csv", "json"]):
    """
    Download ETF holdings data, save in multiple formats, and provide detailed analysis.
    
    Args:
        etf_type: Either "SPY" (S&P 500) or "QQQ" (Nasdaq-100)
        save_formats: List of formats to save ["xlsx", "csv", "json"]
    
    Returns:
        Pandas DataFrame with the cleaned holdings data
    """
    if etf_type.upper() == "SPY":
        # State Street SPDR (SPY) holdings URL
        url = "https://www.ssga.com/us/en/individual/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
        name = "SPY (S&P 500)"
    elif etf_type.upper() == "QQQ":
        # Invesco QQQ holdings URL
        url = "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?audienceType=Investor&action=download&ticker=QQQ"
        name = "QQQ (Nasdaq-100)"
    else:
        raise ValueError("ETF type must be either 'SPY' or 'QQQ'")
    
    print(f"Downloading {name} holdings from: {url}")
    
    try:
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the raw file
        raw_filename = f"{etf_type.lower()}_holdings_raw.xlsx"
        with open(raw_filename, 'wb') as f:
            f.write(response.content)
        print(f"Raw {etf_type} holdings saved to {raw_filename}")
        
        # Analyze file structure before parsing
        print(f"\n{'-'*40}")
        print(f"ANALYZING {etf_type} FILE STRUCTURE")
        print(f"{'-'*40}")
        
        # Try to get Excel file info
        try:
            # Read with minimal parsing to check structure
            xl = pd.ExcelFile(io.BytesIO(response.content))
            sheets = xl.sheet_names
            print(f"Excel file contains {len(sheets)} sheets: {sheets}")
            
            # Inspect each sheet
            for sheet in sheets:
                try:
                    # Try different skip rows values
                    for skip_rows in [0, 1, 2, 3]:
                        try:
                            print(f"\nAttempting to read sheet '{sheet}' with skiprows={skip_rows}:")
                            df_peek = pd.read_excel(io.BytesIO(response.content), sheet_name=sheet, skiprows=skip_rows, nrows=5)
                            print(f"Columns found: {df_peek.columns.tolist()}")
                            print(f"First row sample: {df_peek.iloc[0].to_dict()}")
                            break
                        except Exception as e:
                            print(f"Error with skiprows={skip_rows}: {e}")
                except Exception as sheet_err:
                    print(f"Error inspecting sheet '{sheet}': {sheet_err}")
            
        except Exception as excel_err:
            print(f"Error analyzing Excel structure: {excel_err}")
            
            # Try to read raw bytes and look for patterns
            print("\nAttempting to analyze raw content:")
            try:
                raw_content = response.content.decode('utf-8', errors='replace')
                # Print first few lines or characters
                print(f"First 500 characters: {raw_content[:500]}")
            except:
                print("Could not decode raw content as text")
            
        # Now try our best to parse the data
        print(f"\n{'-'*40}")
        print(f"PARSING {etf_type} HOLDINGS DATA")
        print(f"{'-'*40}")
        
        # Try multiple parsing approaches
        df = None
        parse_success = False
        parse_method = ""
        
        # Approach 1: Try standard Excel parsing with different skiprows
        for skip_rows in [0, 1, 2, 3, 4, 5]:
            try:
                print(f"Trying to parse Excel with skiprows={skip_rows}")
                df = pd.read_excel(io.BytesIO(response.content), skiprows=skip_rows)
                
                # Check if we got sensible data by looking for common column indicators
                col_found = False
                for col in df.columns:
                    if any(keyword in str(col).lower() for keyword in ['ticker', 'symbol', 'security', 'name', 'weight']):
                        col_found = True
                        print(f"Found potential column: {col}")
                
                if col_found and len(df) > 10:  # Sensible number of rows
                    parse_success = True
                    parse_method = f"Excel parsing with skiprows={skip_rows}"
                    break
            except Exception as e:
                print(f"Error with skiprows={skip_rows}: {e}")
        
        # Approach 2: Try all sheets if first approach failed
        if not parse_success and 'sheets' in locals():
            for sheet in sheets:
                for skip_rows in [0, 1, 2, 3]:
                    try:
                        print(f"Trying to parse sheet '{sheet}' with skiprows={skip_rows}")
                        df = pd.read_excel(io.BytesIO(response.content), sheet_name=sheet, skiprows=skip_rows)
                        
                        # Check for sensible data
                        col_found = False
                        for col in df.columns:
                            if any(keyword in str(col).lower() for keyword in ['ticker', 'symbol', 'security', 'name', 'weight']):
                                col_found = True
                                print(f"Found potential column: {col}")
                        
                        if col_found and len(df) > 10:
                            parse_success = True
                            parse_method = f"Excel parsing of sheet '{sheet}' with skiprows={skip_rows}"
                            break
                    except Exception as e:
                        print(f"Error with sheet '{sheet}', skiprows={skip_rows}: {e}")
                
                if parse_success:
                    break
        
        # If we still couldn't parse, try with different engines
        if not parse_success:
            try:
                print("Trying with openpyxl engine")
                df = pd.read_excel(io.BytesIO(response.content), engine='openpyxl')
                parse_success = True
                parse_method = "Excel parsing with openpyxl engine"
            except Exception as e:
                print(f"Error with openpyxl engine: {e}")
        
        if not parse_success:
            print("All parsing attempts failed.")
            return None
        
        print(f"\nSuccessfully parsed {etf_type} holdings using {parse_method}")
        print(f"Data shape: {df.shape}")
        print("\nColumn names:")
        for col in df.columns:
            print(f"- {col}")
        
        # Identify key columns
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
        
        print(f"\nIdentified key columns:")
        print(f"Ticker column: {ticker_col}")
        print(f"Name column: {name_col}")
        print(f"Weight column: {weight_col}")
        
        # Basic cleaning
        if ticker_col:
            # Drop rows with missing tickers
            df = df.dropna(subset=[ticker_col])
            # Clean up ticker symbols (remove whitespace, convert to uppercase)
            df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()
            
        if weight_col:
            # Convert weight to numeric, handling percentage signs and commas
            try:
                # First try directly
                df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
            except:
                # If that fails, try cleaning the data first
                try:
                    # Remove % signs, commas, and convert to float
                    df[weight_col] = df[weight_col].astype(str).str.replace('%', '').str.replace(',', '').astype(float)
                    # If weights are not already in percentage form (i.e., 0.0-100.0), convert them
                    if df[weight_col].max() < 1.0:
                        df[weight_col] = df[weight_col] * 100
                except Exception as e:
                    print(f"Error converting weights to numeric: {e}")
        
        # Save in multiple formats if requested
        today = datetime.now().strftime("%Y-%m-%d")
        
        if "xlsx" in save_formats:
            xlsx_filename = f"{etf_type.lower()}_holdings_{today}.xlsx"
            df.to_excel(xlsx_filename, index=False)
            print(f"Saved {etf_type} holdings to {xlsx_filename}")
        
        if "csv" in save_formats:
            csv_filename = f"{etf_type.lower()}_holdings_{today}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Saved {etf_type} holdings to {csv_filename}")
        
        if "json" in save_formats and ticker_col:
            # Create standardized format for JSON
            standardized_data = []
            
            for i, row in df.iterrows():
                try:
                    item = {}
                    # Add ticker/symbol
                    item["symbol"] = str(row[ticker_col]).strip() if pd.notna(row[ticker_col]) else ""
                    
                    # Add name if available
                    if name_col and pd.notna(row[name_col]):
                        item["name"] = str(row[name_col]).strip()
                    else:
                        item["name"] = f"{item['symbol']} Inc."
                    
                    # Add weight if available
                    if weight_col and pd.notna(row[weight_col]):
                        try:
                            item["weight"] = float(row[weight_col])
                        except:
                            # Use regex to extract numeric part if conversion fails
                            weight_str = str(row[weight_col])
                            weight_match = re.search(r'(\d+\.?\d*)', weight_str)
                            if weight_match:
                                item["weight"] = float(weight_match.group(1))
                    
                    # Add rank
                    item["rank"] = i + 1  # 1-based rank
                    
                    # Add all other non-null fields that might be useful
                    for col in df.columns:
                        if col not in [ticker_col, name_col, weight_col] and pd.notna(row[col]):
                            # Convert to appropriate type
                            val = row[col]
                            if isinstance(val, (int, float)):
                                item[str(col).lower().replace(' ', '_')] = val
                            else:
                                item[str(col).lower().replace(' ', '_')] = str(val).strip()
                    
                    standardized_data.append(item)
                except Exception as e:
                    print(f"Error processing row {i}: {e}")
            
            json_filename = f"{etf_type.lower()}_holdings_{today}.json"
            with open(json_filename, 'w') as f:
                json.dump(standardized_data, f, indent=2)
            print(f"Saved {etf_type} holdings to {json_filename}")
            
            # Print top 10 holdings
            print(f"\nTop 10 {etf_type} holdings:")
            for i, item in enumerate(standardized_data[:10]):
                print(f"{i+1}. {item['symbol']}: {item['name'][:30]:30s} - {item['weight']:.2f}%")
        
        return df
    
    except Exception as e:
        print(f"Error downloading or processing {etf_type} holdings: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_both_etfs():
    """
    Download and analyze both SPY and QQQ ETF holdings data.
    """
    print("="*80)
    print("ANALYZING SPY (S&P 500) HOLDINGS")
    print("="*80)
    spy_df = download_and_analyze_etf_holdings("SPY")
    
    print("\n" + "="*80)
    print("ANALYZING QQQ (NASDAQ-100) HOLDINGS")
    print("="*80)
    qqq_df = download_and_analyze_etf_holdings("QQQ")
    
    return spy_df, qqq_df

# Main function
if __name__ == "__main__":
    analyze_both_etfs()