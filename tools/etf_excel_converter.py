import pandas as pd
import os
import sys
import glob
from pathlib import Path
import json
import datetime
import argparse

def convert_excel_to_csv(input_file, output_file=None, sheet_index=None, skiprows=None):
    """
    Convert an Excel file to CSV with various options to handle different formats.
    
    Args:
        input_file: Path to the Excel file
        output_file: Path to save the CSV file (defaults to same name with .csv extension)
        sheet_index: Index of the sheet to convert (defaults to 0)
        skiprows: Number of rows to skip at the beginning (defaults to None)
        
    Returns:
        Path to the saved CSV file
    """
    try:
        # Set defaults
        if sheet_index is None:
            sheet_index = 0
        
        if output_file is None:
            output_file = Path(input_file).with_suffix('.csv')
        
        print(f"Converting {input_file} to {output_file}")
        print(f"Sheet index: {sheet_index}, Skiprows: {skiprows}")
        
        # First try standard pandas read_excel
        try:
            if skiprows is not None:
                df = pd.read_excel(input_file, sheet_name=sheet_index, skiprows=skiprows)
            else:
                df = pd.read_excel(input_file, sheet_name=sheet_index)
        except Exception as e:
            print(f"Error with standard Excel reading: {e}")
            print("Trying with openpyxl engine...")
            
            if skiprows is not None:
                df = pd.read_excel(input_file, sheet_name=sheet_index, skiprows=skiprows, engine='openpyxl')
            else:
                df = pd.read_excel(input_file, sheet_name=sheet_index, engine='openpyxl')
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Successfully converted to CSV. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Print first few rows
        print("\nFirst 5 rows:")
        print(df.head().to_string())
        
        return output_file
        
    except Exception as e:
        print(f"Error converting {input_file} to CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

def inspect_excel_file(input_file):
    """
    Inspect an Excel file and provide detailed information about its structure.
    
    Args:
        input_file: Path to the Excel file
        
    Returns:
        Dictionary with information about the Excel file
    """
    try:
        print(f"Inspecting Excel file: {input_file}")
        
        # Get basic file info
        file_size = os.path.getsize(input_file) / 1024  # KB
        file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(input_file))
        
        print(f"File size: {file_size:.2f} KB")
        print(f"Last modified: {file_modified}")
        
        # Try to get Excel file structure with both engines
        results = {"file_path": str(input_file),
                   "file_size_kb": file_size,
                   "last_modified": file_modified.isoformat(),
                   "sheets": []}
        
        engines = ['standard', 'openpyxl']
        for engine in engines:
            try:
                print(f"\nTrying with {engine} engine:")
                
                if engine == 'standard':
                    xl = pd.ExcelFile(input_file)
                else:
                    xl = pd.ExcelFile(input_file, engine='openpyxl')
                    
                sheets = xl.sheet_names
                print(f"Found {len(sheets)} sheets: {sheets}")
                
                sheets_info = []
                
                # Inspect each sheet
                for sheet_name in sheets:
                    sheet_info = {"name": sheet_name, "rows": []}
                    
                    # Try different skiprows values
                    for skiprows in [0, 1, 2, 3]:
                        try:
                            if engine == 'standard':
                                df = pd.read_excel(input_file, sheet_name=sheet_name, skiprows=skiprows, nrows=5)
                            else:
                                df = pd.read_excel(input_file, sheet_name=sheet_name, skiprows=skiprows, nrows=5, engine='openpyxl')
                                
                            row_info = {
                                "skiprows": skiprows,
                                "shape": df.shape,
                                "columns": df.columns.tolist()
                            }
                            
                            sheet_info["rows"].append(row_info)
                            
                            print(f"\nSheet '{sheet_name}' with skiprows={skiprows}:")
                            print(f"  Shape: {df.shape}")
                            print(f"  Columns: {df.columns.tolist()}")
                            
                            # Check if we see potential index columns
                            for col in df.columns:
                                col_lower = str(col).lower()
                                if ('ticker' in col_lower or 'symbol' in col_lower or 
                                    'name' in col_lower or 'weight' in col_lower):
                                    print(f"  Potential index column: {col}")
                            
                            # Print first row sample
                            try:
                                if len(df) > 0:
                                    print(f"  First row sample: {df.iloc[0].to_dict()}")
                            except:
                                pass
                                
                        except Exception as e:
                            print(f"  Error with skiprows={skiprows}: {e}")
                    
                    sheets_info.append(sheet_info)
                
                results["sheets"].extend(sheets_info)
                
                # We succeeded with this engine, no need to try others
                results["engine_used"] = engine
                break
                
            except Exception as e:
                print(f"Error inspecting with {engine} engine: {e}")
                continue
        
        # Save inspection results to JSON
        output_json = f"{Path(input_file).stem}_inspection.json"
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nInspection results saved to {output_json}")
        
        return results
        
    except Exception as e:
        print(f"Error inspecting {input_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_convert_directory(directory, pattern="*.xlsx"):
    """
    Convert all Excel files in a directory to CSV.
    
    Args:
        directory: Directory containing Excel files
        pattern: File pattern to match (default: *.xlsx)
    """
    try:
        print(f"Batch converting Excel files in {directory} matching {pattern}")
        
        # Find all matching files
        if isinstance(directory, str):
            directory = Path(directory)
            
        excel_files = list(directory.glob(pattern))
        
        if not excel_files:
            print(f"No files found matching {pattern} in {directory}")
            return
            
        print(f"Found {len(excel_files)} files to convert")
        
        results = []
        
        for excel_file in excel_files:
            print(f"\n{'='*50}")
            print(f"Processing: {excel_file}")
            
            # First inspect the file
            inspect_result = inspect_excel_file(excel_file)
            
            # Try to determine the best parameters for conversion
            skiprows = 0
            sheet_index = 0
            
            if inspect_result and "sheets" in inspect_result and inspect_result["sheets"]:
                # Use the first sheet by default
                sheet = inspect_result["sheets"][0]
                
                # Try to find the skiprows value that gives us index-related columns
                best_skiprows = None
                
                for row_info in sheet.get("rows", []):
                    skiprows_val = row_info.get("skiprows", 0)
                    columns = row_info.get("columns", [])
                    
                    # Check if any column looks like what we want
                    col_match = False
                    for col in columns:
                        col_lower = str(col).lower()
                        if ('ticker' in col_lower or 'symbol' in col_lower or 
                            'name' in col_lower or 'weight' in col_lower):
                            col_match = True
                            break
                    
                    if col_match:
                        best_skiprows = skiprows_val
                        break
                
                # If we found a good skiprows value, use it
                if best_skiprows is not None:
                    skiprows = best_skiprows
            
            # Convert the file with the determined parameters
            output_file = excel_file.with_suffix('.csv')
            result = convert_excel_to_csv(excel_file, output_file, sheet_index=sheet_index, skiprows=skiprows)
            
            if result:
                results.append({"input": str(excel_file), "output": str(result), "success": True})
            else:
                results.append({"input": str(excel_file), "success": False})
        
        # Save batch results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_results_file = directory / f"batch_conversion_results_{timestamp}.json"
        
        with open(batch_results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nBatch conversion completed. Results saved to {batch_results_file}")
        print(f"Successfully converted: {sum(1 for r in results if r.get('success', False))}/{len(results)} files")
        
        return results
        
    except Exception as e:
        print(f"Error in batch conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='ETF Excel File Converter and Inspector')
    parser.add_argument('--mode', choices=['convert', 'inspect', 'batch'], default='inspect',
                       help='Operation mode: convert a single file, inspect a file, or batch convert a directory')
    parser.add_argument('--input', required=True, help='Input file or directory path')
    parser.add_argument('--output', help='Output file (for single conversion)')
    parser.add_argument('--sheet', type=int, default=0, help='Sheet index to use (default: 0)')
    parser.add_argument('--skiprows', type=int, help='Number of rows to skip (default: auto-detect)')
    parser.add_argument('--pattern', default='*.xlsx', help='File pattern for batch mode (default: *.xlsx)')
    
    args = parser.parse_args()
    
    if args.mode == 'convert':
        convert_excel_to_csv(args.input, args.output, args.sheet, args.skiprows)
    elif args.mode == 'inspect':
        inspect_excel_file(args.input)
    elif args.mode == 'batch':
        batch_convert_directory(args.input, args.pattern)
    else:
        print("Invalid mode specified")

if __name__ == "__main__":
    main()