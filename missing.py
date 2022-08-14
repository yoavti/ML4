import os

import pandas as pd

from experiment_utils.parameters import ks
from experiment_utils.parameters import named_score_funcs


def add_dict_row(d, dataset, k=None, files=None):
    d['dataset'].append(dataset)
    d['k'].append(k)
    d['files'].append(files)


def find_missing():
    missing_dict = dict(dataset=[], k=[], files=[])
    fss = list(named_score_funcs) + ['SelectFdr']
    results_path = 'results'
    with open('datasets.txt', 'r') as f:
        lines = f.readlines()
    for dataset in lines:
        dataset = dataset.strip()
        dataset_results_path = os.path.join(results_path, dataset)
        if not os.path.exists(dataset_results_path):
            add_dict_row(missing_dict, dataset)
            continue
        for k in ks:
            k_results_path = os.path.join(dataset_results_path, str(k))
            if not os.path.exists(k_results_path):
                add_dict_row(missing_dict, dataset, k)
                continue
            files = ['cv_results.csv', 'fs.csv', 'best.json']
            fs_files = [f'{fs}.csv' for fs in fss]
            files += fs_files
            missing_files = [file for file in files if not os.path.exists(os.path.join(k_results_path, file))]
            if missing_files:
                add_dict_row(missing_dict, dataset, k, missing_files)
    missing_df = pd.DataFrame(missing_dict)
    missing_df.to_csv('missing.csv', index=False)


if __name__ == '__main__':
    find_missing()
