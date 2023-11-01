#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
https://medium.com/@jb.ranchana/write-and-append-dataframes-to-google-sheets-in-python-f62479460cf0
"""

import pandas as pd
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

# open a google sheet
gs = gc.open_by_key('1BSlmpq9yu3xU13cR0uqjoJfWW8ADm81zIXCVhYPEjj4')

worksheet_list = gs.worksheets()
print(f'existing worksheets: {worksheet_list}')

if __name__ == '__main__':

    import pandas as pd

    # dataframe (create or import it)
    df = pd.DataFrame({'a': ['apple','airplane','alligator'], 'b': ['banana', 'ball', 'butterfly'], 'c': ['cantaloupe', 'crane', 'cat']})
    
    # write to dataframe
    ws = gs.add_worksheet(title="test sheet", rows=100, cols=20)

    set_with_dataframe(
        worksheet=ws, 
        dataframe=df, 
        include_index=False,
        include_column_header=True, 
        resize=True,
        )