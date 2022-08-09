import os

import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('best.csv')
    for _, row in df.iterrows():
        col_args_pairs = [('dataset', 'd'), ('fs', 'fs'), ('clf', 'clf'), ('k', 'k')]
        command = 'aug_experiments.py'
        for col, arg in col_args_pairs:
            command += f' -{arg} {row[col]}'
        print(command)
        os.system(command)
