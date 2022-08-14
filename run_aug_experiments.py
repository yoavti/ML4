import os

import pandas as pd


def run_aug_experiments():
    df = pd.read_csv('best.csv')
    for _, row in df.iterrows():
        col_args_pairs = [('dataset', 'd'), ('fs', 'fs'), ('clf', 'clf'), ('k', 'k')]
        command = 'D:\\anaconda3\\envs\\ML4\\python.exe aug_experiments.py'
        for col, arg in col_args_pairs:
            command += f' -{arg} {row[col]}'
        print(command)
        os.system(command)


if __name__ == '__main__':
    run_aug_experiments()
