import os

import pandas as pd

from data.experiments import ARFF, bioconductor, Datamicroarray, scikit_feature_datasets
from experiment_utils.parameters import ks


def func_to_name(func):
    return func[10:-19]


def until(s, c):
    idx = s.find(c)
    s = s[:idx]
    return s


def extract_function_name(func):
    return func[10:-19]


def stringify_score_func(s):
    while s.contains('<function ') and s.contains(' at 0x' and s.contains('>')):
        start_idx = s.find('<function ')
        end_idx = s.find('>')
        func = s[start_idx:end_idx+1]
        func_name = extract_function_name(func)
        s = s.replace(func, func_name)
    return s


def read_cv_results(path):
    df = pd.read_csv(path)
    df = df.drop('Unnamed: 0', axis=1)

    df['param_fs__transformer'] = df['param_fs__transformer'].apply(stringify_score_func)
    return df


def gather_scores(metric='ROC_AUC'):
    results_path = 'results'
    directories = [ARFF, bioconductor, Datamicroarray, scikit_feature_datasets]
    for directory in directories:
        for dataset in directory.load.datasets:
            dataset_results_path = os.path.join(results_path, dataset)
            if not os.path.exists(dataset_results_path):
                continue
            for k in ks:
                k_results_path = os.path.join(dataset_results_path, str(k))
                if not os.path.exists(k_results_path):
                    continue
                cv_results_path = os.path.join(k_results_path, 'cv_results.csv')
                if not os.path.exists(cv_results_path):
                    continue
                cv_results = read_cv_results(cv_results_path)
                for _, cv_row in cv_results.iterrows():
                    transformer = cv_row['param_fs__transformer']
                    print(transformer)
                    # score = cv_row[f'mean_test_{metric}']


if __name__ == '__main__':
    gather_scores()
