#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd

from open_lid import LIDModel

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input file')
    parser.add_argument('-o', '--output_file', type=str, default=None, help='Output file')
    parser.add_argument('--keys', nargs='+', default=['source', 'system'], help='Keys to run language assignment on')
    args = parser.parse_args()
    return args

def main(args):
    
    data = pd.read_json(args.input_file, lines=True)

    lid = LIDModel()

    for key in args.keys:
        if key in data.columns:
            data[f'{key}_lang'] = data[key].apply(lambda x: lid.predict(x)[0])
        else:
            print(f'{key} not in data')

    if args.output_file:
        data.to_json(args.output_file, lines=True, orient='records', force_ascii=False)
    else:
        data.to_json(args.input_file, lines=True, orient='records', force_ascii=False)

    print(f'Finished assigning languages to {args.input_file}')

if __name__ == '__main__':
    args = set_args()
    main(args)