#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Gathers data from multiple annotation instances files and outputs a google sheet for inspection.

Generates 1 spreadsheet for each language (based on the input files specified). 

Example usage:

    # gather de outputs from 3 models (1 without tgt lang, 2 with tgt lang)
    python make_comp_gsheet_v2.py \
        -i data/to_annotate/llama_2_7b_hf_de_merged-alpaca_eval_instructions_de-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_fr_merged-alpaca_eval_instructions_de-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_zh_merged-alpaca_eval_instructions_de-anno_instances.jsonl \
        --fields "source" "system_0" "system_42" "system_723" -n 20

    python make_comp_gsheet_v2.py \
        -i data/to_annotate/llama_2_7b_hf_es_merged-alpaca_eval_instructions_es-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_fr_merged-alpaca_eval_instructions_es-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_zh_merged-alpaca_eval_instructions_es-anno_instances.jsonl \
        --fields "source" "system_0" "system_42" "system_723" -n 20
    
    python make_comp_gsheet_v2.py \
        -i data/to_annotate/llama_2_7b_hf_de_merged-alpaca_eval_instructions_en-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_fr_merged-alpaca_eval_instructions_en-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_zh_merged-alpaca_eval_instructions_en-anno_instances.jsonl \
        --fields "source" "system_0" "system_42" "system_723" -n 20

    # norwegian
    python make_comp_gsheet_v2.py \
        -i data/to_annotate/llama_2_7b_hf_de_merged-alpaca_eval_instructions_no-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_fr_merged-alpaca_eval_instructions_no-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_zh_merged-alpaca_eval_instructions_no-anno_instances.jsonl \
        --fields "source" "system_0" "system_42" "system_723" -n 20

    python make_comp_gsheet_v2.py \
        -i data/to_annotate/llama_2_7b_hf_de_merged-alpaca_eval_instructions_is-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_fr_merged-alpaca_eval_instructions_is-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_ru_merged-alpaca_eval_instructions_is-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_zh_merged-alpaca_eval_instructions_is-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_es_merged-alpaca_eval_instructions_is-anno_instances.jsonl \
        data/to_annotate/llama_2_7b_hf_ca_merged-alpaca_eval_instructions_is-anno_instances.jsonl \
        --fields "source" "system_0" "system_42" "system_723" -n 20

"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


scopes = ['https://www.googleapis.com/auth/spreadsheets',
          'https://www.googleapis.com/auth/drive']

credentials = Credentials.from_service_account_file('private/ml-llm-403314-c9d09e511fb6.json', scopes=scopes)

gc = gspread.authorize(credentials)

gauth = GoogleAuth()
drive = GoogleDrive(gauth)

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_files", required=True, nargs='+', help="path to files")
    ap.add_argument("--fields", required=True, nargs='+', help="fields to include in output")
    ap.add_argument("-n", "--num_instances", required=False, default=-1, type=int, help="number of instances to include in output")
    # ap.add_argument("--lang", required=False, help="language tag of the input file")
    return ap.parse_args()


if __name__ == "__main__":
    args = set_args()

    # Read in data
    dfs = []
    keys = []
    langs = []
    for input_file in args.input_files:
        keys.append(Path(input_file).stem.split('-')[0])
        langs.append(Path(input_file).stem.split('-')[1].split('_')[-1])
        dfs.append(pd.read_json(input_file, lines=True))
    
    df = pd.concat(dfs, axis=1, keys=keys)

    if args.fields:
        idx = pd.IndexSlice
        df = df.loc[:, idx[:, args.fields]]

    if args.num_instances > 0:
        df = df.sample(n=min(args.num_instances, len(df)), random_state=42)

    print(f'Gathered {len(df)} instances from {len(args.input_files)} files.')
    print(f'Models: {keys}')
    print(f'Test sets: {langs}')

    assert len(set(langs)) == 1, "All input files must be for the same language."
    
    lang_tag = langs[0]


    # open existing google sheet
    gs = gc.open_by_key('1BSlmpq9yu3xU13cR0uqjoJfWW8ADm81zIXCVhYPEjj4')

    worksheet_list = gs.worksheets()
    print(f'existing worksheets: {worksheet_list}')
    
    # create a new google sheet
    timestamp = datetime.now().strftime("%Y-%m-%d")

    sheet_name = f"{lang_tag}-{timestamp}"
    try:
        worksheet = gs.add_worksheet(title=sheet_name, rows=len(df), cols=len(df.columns))
    except:
        worksheet = gs.worksheet(sheet_name)
        worksheet.clear() # Clear the worksheet before appending data
        worksheet.resize(rows=len(df), cols=len(df.columns))

    print(f'new worksheet: {worksheet}')

    # send the dataframe to the new google sheet
    set_with_dataframe(
        worksheet=worksheet, 
        dataframe=df, 
        include_index=False,
        include_column_header=True, 
        resize=True,
        )

    print(f'wrote {len(df)} rows to {sheet_name}.')    
             






