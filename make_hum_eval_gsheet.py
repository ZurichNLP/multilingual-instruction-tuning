#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example usage:

    python make_hum_eval_gsheet.py \
        -i data/to_annotate/llama_2_7b_hf_ml2_merged-alpaca_eval_instructions_de-anno_instances.jsonl

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
    ap.add_argument("-i", "--input_file", required=True, help="path to file")
    ap.add_argument("--fields", required=True, nargs='+', help="fields to include in output")
    ap.add_argument("--users", required=True, nargs='+', help="users to add to google sheet")
    return ap.parse_args()

if __name__ == "__main__":
    args = set_args()

    # Read in data
    df = pd.read_json(args.input_file, lines=True)

    
    if args.fields:
        df = df[args.fields]

    # add criteria columns
    df['relevance'] = np.nan
    df['fluency'] = np.nan
    df['naturalness'] = np.nan
    df['overall'] = np.nan

    # get the language tag of the input file
    lang_tag = Path(args.input_file).stem.split('-')[1].split('_')[-1]

    # create a new google sheet
    timestamp = datetime.now().strftime("%Y-%m-%d")
    spreadsheet = gc.create(f'ml-llm hum eval {lang_tag} - {timestamp}')

    # share the new google sheet with the ml-llm team
    for user in args.users:
        spreadsheet.share(user, perm_type='user', role='writer')
        print(f'{spreadsheet.id} invitation sent to {user}')

    # get worksheet of the new google sheet
    worksheet = gc.open_by_key(spreadsheet.id).sheet1

    # send the dataframe to the new google sheet
    set_with_dataframe(
        worksheet=worksheet, 
        dataframe=df, 
        include_index=False,
        include_column_header=True, 
        resize=True,
        )
    
             






