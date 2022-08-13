import os

import numpy as np
import pandas as pd

from functools import partial
from pprint import PrettyPrinter

from data.experiments import ARFF, bioconductor, Datamicroarray, scikit_feature_datasets
from experiment_utils.parameters import ks

pp = PrettyPrinter()


def func_to_name(func):
    return func[10:-19]


def extract_function_name(func):
    return func[10:-19]


def stringify_score_func(s):
    while all(substr in s for substr in ['<function ', ' at 0x', '>']):
        start_idx = s.find('<function ')
        end_idx = s.find('>')
        func = s[start_idx:end_idx+1]
        func_name = extract_function_name(func)
        s = s.replace(func, func_name)
    return s


def until(s, c):
    idx = s.find(c)
    s = s[:idx]
    return s


def read_cv_results(path):
    df = pd.read_csv(path)
    df = df.drop('Unnamed: 0', axis=1)

    df['param_fs__transformer'] = df['param_fs__transformer'].apply(stringify_score_func)

    df['param_clf__estimator'] = df['param_clf__estimator'].astype(str)
    df['param_clf__estimator'] = df['param_clf__estimator'].apply(partial(until, c='()'))
    return df


def gather_scores(metric='ROC_AUC'):
    scores_dict = {}
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
                    clf = cv_row['param_clf__estimator']
                    meta_clf = f'{transformer}->{clf}'
                    score = cv_row[f'mean_test_{metric}']
                    if np.isnan(score):
                        continue
                    if dataset not in scores_dict:
                        scores_dict[dataset] = {}
                    scores_dict[dataset][meta_clf] = score
    return scores_dict


def keep_shared_classifiers(scores_dict):
    sorted_datasets = sorted(list(scores_dict), key=lambda dataset: len(scores_dict[dataset]), reverse=True)
    chosen_datasets = set()
    chosen_classifiers = set()
    for dataset in sorted_datasets:
        classifiers_to_add = set(scores_dict[dataset])
        if not chosen_datasets:
            chosen_datasets.add(dataset)
            chosen_classifiers = classifiers_to_add
        classifiers_to_add = classifiers_to_add
        new_intersection = chosen_classifiers.intersection(classifiers_to_add)
        if not new_intersection:
            break
        chosen_datasets.add(dataset)
        chosen_classifiers = new_intersection
    scores_dict = {dataset: {classifier: score
                             for classifier, score in classifiers.items()
                             if classifier in chosen_classifiers}
                   for dataset, classifiers in scores_dict.items()
                   if dataset in chosen_datasets}
    return scores_dict


def main():
    scores_dict = gather_scores()
    scores_dict = keep_shared_classifiers(scores_dict)


if __name__ == '__main__':
    main()
