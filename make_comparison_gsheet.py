#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Script to extract items from multiple (line-aligned) jsonl files and uploading to Google Sheets for inspection.

Input jsonl files must be line-aligned!

Example usage:

    python make_comparison_gsheet.py data/alpaca_eval_instructions_*.json

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
    return ap.parse_args()

def load_data(files):
    dfs = []

    if len(files) == 0:
        raise ValueError("No input files provided")

    for f in files:
        print(f"Loading {f}")
        df = pd.read_json(f, lines=True)
        name_map = {c: f'{Path(f).name} {c}' for c in df.columns}
        df = df.rename(columns=name_map)
        df.name = f
        dfs.append(df)
    
    # merge dataframes on source column
    df = pd.concat(dfs, axis=1)
        
    # remove duplicate columns (e.g. source, reference)
    df = df.loc[:,~df.columns.duplicated()].copy()
    
    # reorder columns
    if "source" in df.columns and "reference" in df.columns:
        df = df[["source", "reference"] + [c for c in df.columns if c not in ["source", "reference"]]]

    return df

if __name__ == "__main__":
    args = set_args()
    df = load_data(args.files)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    sheet_name = f"{args.sheet_name}-{timestamp}" if args.sheet_name else timestamp

    try:
        ws = gs.add_worksheet(title=sheet_name, rows=df.shape[0], cols=df.shape[1])
    except:
        ws = gs.worksheet(sheet_name)

    set_with_dataframe(
        worksheet=ws, 
        dataframe=df, 
        include_index=False,
        include_column_header=True, 
        resize=True,
        )