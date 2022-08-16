import os
import json

import pandas as pd

from data import data_loader
from experiment_utils.metrics import get_metrics
from experiment_utils.parameters import named_classifiers, ks
from results_processing_utils.read_csv import read_cv_results


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def aug():
    create_if_not_exists('aug')
    results_path = 'results'
    metric_names = list(get_metrics())
    columns_mapping = {f'mean_test_{metric}': metric for metric in metric_names}
    for time in ['fit', 'score']:
        columns_mapping[f'mean_{time}_time'] = f'{time}_time'
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
        comparison_df.to_csv(os.path.join('aug', f'{dataset}.csv'))


def improvement():
    create_if_not_exists('improvement')
    results_path = 'results'
    metric_names = list(get_metrics())
    columns_mapping = {f'mean_test_{metric}': metric for metric in metric_names}
    columns_mapping['mean_fit_time'] = 'time'
    param_columns = ['param_fs__transformer', 'param_fs__transformer__score_func', 'param_clf__estimator']
    for dataset in data_loader.available_datasets():
        dataset_improvement_path = os.path.join('improvement', dataset)
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
            create_if_not_exists(dataset_improvement_path)
            cv_results = read_cv_results(cv_results_path)
            cv_results = cv_results[list(columns_mapping) + param_columns]
            cv_results = cv_results[cv_results['param_fs__transformer'] == 'SelectKBest']
            cv_results = cv_results.drop('param_fs__transformer', axis=1)
            for classifier in named_classifiers:
                clf_results = cv_results[cv_results['param_clf__estimator'] == classifier]
                clf_results = clf_results.drop('param_clf__estimator', axis=1)
                comparison_dict = {}
                for score_func in ['ufs_sp_l_2_1', 'ufs_sp_f']:
                    score_results = clf_results[clf_results['param_fs__transformer__score_func'] == score_func]
                    score_results = score_results.drop('param_fs__transformer__score_func', axis=1)
                    score_results = score_results.rename(columns_mapping, axis=1)
                    score_results = score_results.mean()
                    comparison_dict[score_func] = score_results
                comparison_df = pd.DataFrame(comparison_dict)
                comparison_df.to_csv(os.path.join(dataset_improvement_path, f'{classifier}.csv'))


if __name__ == '__main__':
    aug()
    improvement()
