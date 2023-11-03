#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Script to extract items from multiple (line-aligned) jsonl files and uploading to Google Sheets for inspection.

Input jsonl files must be line-aligned!

Example usage:

    python make_comparison_gsheet.py data/alpaca_eval_instructions_*.json --format_type inputs

    python make_comparison_gsheet.py data/outputs/llama_2_7b_hf_de_merged/alpaca_eval_instructions_de-none-guanaco_prompt-s*.jsonl --format_type outputs

TODO:
- handle errors for existing sheet names

"""

import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
from sheets import *

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs='+', help="paths to input jsonl files")

    ap.add_argument("--outfile", required=False, default=None, help="path to output file")
    ap.add_argument("-c", "--columns", required=False, nargs="+", default=None, help="name of columns to extract")
    ap.add_argument("-s", "--sheet_name", required=False, default=None, help="name of sheet to create")
    ap.add_argument("--format_type", default=None, help="type of format to extract (outputs, inputs)")
    return ap.parse_args()

def load_data(files):
    dfs = []

    if len(files) == 0:
        raise ValueError("No input files provided")

    for f in files:
        print(f"Loading {f}")
        df = pd.read_json(f, lines=True)
        # reorder so that source column is first
        if "source" in df.columns:
            df = df[["source"] + df.columns.drop("source").tolist()]

        name_map = {c: f'{Path(f).name} {c}' for c in df.columns}
        df = df.rename(columns=name_map)
        df.name = f
        dfs.append(df)
    
    # merge dataframes on source column
    df = pd.concat(dfs, axis=1)
    
    # remove columns with names ending with ["finish_reason", "secs", "prompt", "source_lang", "system_lang"]
    for col in df.columns:
        if col.endswith(("finish_reason", "secs", "prompt", "source_lang", "system_lang")):
            df.drop(col, axis=1, inplace=True)

    # remove columns which are duplicated, keeping the first
    df = df[df.columns[~df.T.duplicated()]]

    cols = df.columns.tolist()
    print(f"Loaded {len(cols)} columns: {cols}")


    return df

if __name__ == "__main__":
    args = set_args()
    df = load_data(args.files)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    sheet_name = f"{args.sheet_name}-{timestamp}" if args.sheet_name else timestamp

    try:
        worksheet = gs.add_worksheet(title=sheet_name, rows=df.shape[0], cols=df.shape[1])
    except:
        worksheet = gs.worksheet(sheet_name)
        worksheet.clear() # Clear the worksheet before appending data
        worksheet.resize(rows=len(df), cols=len(df.columns))

    try:
        set_with_dataframe(
            worksheet=worksheet, 
            dataframe=df, 
            include_index=False,
            include_column_header=True, 
            resize=True,
            )
    except gspread.exceptions.APIError: # if the sheet is too large, try uploading in chunks
        
        chunk_size = 50
        
        # Resize the worksheet to accommodate the dataframe and header
        worksheet.resize(rows=len(df) + 1, cols=len(df.columns))
        
        # Write the DataFrame's header
        header_values = df.columns.tolist()
        worksheet.update('A1', [header_values])

        # Iterate over the DataFrame and update cells in chunks
        for start in range(0, len(df), chunk_size):
            # Calculate end row
            end = min(start + chunk_size, len(df))
            chunk = df.iloc[start:end]
            
            # Calculate the cell range to update
            start_cell = f'A{start + 2}' # +2 to account for header and zero-index
            end_cell = gspread.utils.rowcol_to_a1(end + 1, len(df.columns))  # +1 to account for header
            
            # Update in batch
            try:
                worksheet.update(f'{start_cell}:{end_cell}', chunk.values.tolist())
                print(f'Uploaded rows {start + 2} to {end + 1}')
            except Exception as e:
                print(f'An error occurred: {e}')