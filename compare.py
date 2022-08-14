import os
import json

import pandas as pd

from functools import partial

from data import data_loader
from experiment_utils.metrics import get_metrics
from experiment_utils.parameters import named_classifiers, ks


def func_to_name(func):
    return func[10:-19]


def until(s, c):
    idx = s.find(c)
    s = s[:idx]
    return s


def read_cv_results(path):
    df = pd.read_csv(path)
    df = df.drop('Unnamed: 0', axis=1)

    df['param_fs__transformer__score_func'] = df['param_fs__transformer__score_func'].astype(str)
    df['param_fs__transformer__score_func'] = df['param_fs__transformer__score_func'].apply(func_to_name)

    df['param_fs__transformer'] = df['param_fs__transformer'].astype(str)
    df['param_fs__transformer'] = df['param_fs__transformer'].apply(partial(until, c='('))

    df['param_clf__estimator'] = df['param_clf__estimator'].astype(str)
    df['param_clf__estimator'] = df['param_clf__estimator'].apply(partial(until, c='()'))
    return df


def aug():
    res_dict = {}
    results_path = 'results'
    metric_names = list(get_metrics())
    columns_mapping = {f'mean_test_{metric}': metric for metric in metric_names}
    columns_mapping['mean_fit_time'] = 'time'
    for dataset in data_loader.available_datasets():
        dataset_results_path = os.path.join(results_path, dataset)
        if not os.path.exists(dataset_results_path):
            continue
        aug_path = os.path.join(dataset_results_path, 'aug')
        if not os.path.exists(aug_path):
            continue
        aug_results_path = os.path.join(aug_path, 'results.csv')
        if not os.path.exists(aug_results_path):
            continue
        aug_results = pd.read_csv(aug_results_path)
        aug_results = aug_results.mean()
        aug_parameters_path = os.path.join(aug_path, 'parameters.json')
        if not os.path.exists(aug_parameters_path):
            continue
        with open(aug_parameters_path, 'r') as f:
            aug_parameters = json.load(f)
        fs = aug_parameters['fs']
        clf = aug_parameters['clf']
        k = aug_parameters['k']
        k_results_path = os.path.join(dataset_results_path, str(k))
        if not os.path.exists(k_results_path):
            continue
        cv_results_path = os.path.join(k_results_path, 'cv_results.csv')
        if not os.path.exists(cv_results_path):
            continue
        cv_results = read_cv_results(cv_results_path)
        if fs == 'SelectFdr':
            cv_results = cv_results[cv_results['param_fs__transformer'] == 'SelectFdr']
        else:
            cv_results = cv_results[cv_results['param_fs__transformer__score_func'] == fs]
        cv_results = cv_results[cv_results['param_clf__estimator'] == clf]
        cv_results = cv_results[list(columns_mapping)]
        cv_results = cv_results.rename(columns_mapping, axis=1)
        cv_results = cv_results.mean()
        comparison_dict = dict(aug=aug_results, original=cv_results)
        comparison_df = pd.DataFrame(comparison_dict)
        res_dict[dataset] = comparison_df
    return res_dict


def improvement():
    res_dict = {}
    results_path = 'results'
    metric_names = list(get_metrics())
    columns_mapping = {f'mean_test_{metric}': metric for metric in metric_names}
    columns_mapping['mean_fit_time'] = 'time'
    param_columns = ['param_fs__transformer', 'param_fs__transformer__score_func', 'param_clf__estimator']
    for dataset in data_loader.available_datasets():
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
            fs_path = os.path.join(k_results_path, 'fs.csv')
            if not os.path.exists(fs_path):
                continue
            cv_results = read_cv_results(cv_results_path)
            cv_results = cv_results[list(columns_mapping) + param_columns]
            cv_results = cv_results[cv_results['param_fs__transformer'] == 'SelectKBest']
            cv_results = cv_results.drop('param_fs__transformer', axis=1)
            comparison_dict = {}
            for score_func in ['ufs_sp_l_2_1', 'ufs_sp_f']:
                score_cv_results = cv_results[cv_results['param_fs__transformer__score_func'] == score_func]
                score_cv_results = score_cv_results.drop('param_fs__transformer__score_func', axis=1)
                for classifier in named_classifiers:
                    meta_clf = f'{score_func}->{classifier}'
                    classifier_cv_results = score_cv_results[score_cv_results['param_clf__estimator'] == classifier]
                    classifier_cv_results = classifier_cv_results.drop('param_clf__estimator', axis=1)
                    classifier_cv_results = classifier_cv_results.rename(columns_mapping, axis=1)
                    classifier_cv_results = classifier_cv_results.mean()
                    comparison_dict[meta_clf] = classifier_cv_results
            comparison_df = pd.DataFrame(comparison_dict)
            res_dict[dataset] = comparison_df
    return res_dict


def save_results(func):
    for dataset, df in func().items():
        df.to_csv(f'{func.__name__}_{dataset}.csv')


def main():
    save_results(aug)
    save_results(improvement)


if __name__ == '__main__':
    main()
