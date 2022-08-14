import os
import json

import pandas as pd

from data import data_loader
from experiment_utils.parameters import ks
from experiment_utils.cv import num_rows, num_folds, cv_method_name
from experiment_utils.metrics import get_metrics
from results_processing_utils.read_csv import read_cv_results, read_fs


def add_expr_row(dictionary, dataset, n, d, fs, clf, k, cv, fold, metric_name, metric_value, features, scores):
    dictionary['Dataset Name'].append(dataset)
    dictionary['Number of Samples'].append(n)
    dictionary['Original Number of Features'].append(d)
    dictionary['Filtering Algorithm'].append(fs)
    dictionary['Learning Algorithm'].append(clf)
    dictionary['Number of Features Selected'].append(k)
    dictionary['CV Method'].append(cv)
    dictionary['Fold'].append(fold)
    dictionary['Measure Type'].append(metric_name)
    dictionary['Measure Value'].append(metric_value)
    dictionary['List of Selected Feature Names'].append(features)
    dictionary['Selected Feature Scores'].append(scores)


def fs_method_name(transformer, score_func):
    if score_func:
        return score_func
    return transformer


def aggregate_results():
    res_dict = {'Dataset Name': [], 'Number of Samples': [], 'Original Number of Features': [],
                'Filtering Algorithm': [], 'Learning Algorithm': [], 'Number of Features Selected': [], 'CV Method': [],
                'Fold': [], 'Measure Type': [], 'Measure Value': [], 'List of Selected Feature Names': [],
                'Selected Feature Scores': []}
    results_path = 'results'
    metric_names = list(get_metrics())
    for dataset in data_loader.available_datasets():
        d, n = data_loader.dataset_size(dataset)
        dataset_results_path = os.path.join(results_path, dataset)
        if not os.path.exists(dataset_results_path):
            continue
        n = num_rows(n)
        _num_folds = num_folds(n)
        _cv_method_name = cv_method_name(n)
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
            fs = read_fs(fs_path)
            fs_idx = 0
            for _, cv_row in cv_results.iterrows():
                transformer = cv_row['param_fs__transformer']
                score_func = cv_row['param_fs__transformer__score_func']
                fs_name = fs_method_name(transformer, score_func)
                clf = cv_row['param_clf__estimator']
                for fold in range(_num_folds):
                    fs_row = fs.iloc[fs_idx]
                    fs_scores = fs_row['scores']
                    fs_features = fs_row['features']
                    for metric_name in metric_names:
                        metric_col_name = f'split{fold}_test_{metric_name}'
                        metric_value = cv_row[metric_col_name]
                        add_expr_row(res_dict, dataset, n, d, fs_name, clf, k, _cv_method_name, fold, metric_name,
                                     metric_value, fs_features, fs_scores)
                    fs_time = fs_row['times']
                    add_expr_row(res_dict, dataset, n, d, fs_name, clf, k, _cv_method_name, fold, 'fs_time',
                                 fs_time, fs_features, fs_scores)
                    fs_idx = (fs_idx + 1) % fs.shape[0]
                mean_fit_time = cv_row['mean_fit_time']
                add_expr_row(res_dict, dataset, n, d, fs_name, clf, k, _cv_method_name, 'N/A', 'mean_fit_time',
                             mean_fit_time, 'N/A', 'N/A')
        aug_path = os.path.join(dataset_results_path, 'aug')
        if not os.path.exists(aug_path):
            continue
        aug_results_path = os.path.join(aug_path, 'results.csv')
        if not os.path.exists(aug_results_path):
            continue
        aug_results = pd.read_csv(aug_results_path)
        aug_parameters_path = os.path.join(aug_path, 'parameters.json')
        if not os.path.exists(aug_parameters_path):
            continue
        with open(aug_parameters_path, 'r') as f:
            aug_parameters = json.load(f)
        aug_fs = aug_parameters['fs'] + '(aug)'
        aug_clf = aug_parameters['clf']
        aug_k = aug_parameters['k']
        for _, row in aug_results.iterrows():
            for fold in range(_num_folds):
                for metric_name in metric_names:
                    metric_value = row[metric_name]
                    add_expr_row(res_dict, dataset, n, d, aug_fs, aug_clf, aug_k, _cv_method_name, fold,
                                 metric_name, metric_value, 'N/A', 'N/A')
                add_expr_row(res_dict, dataset, n, d, aug_fs, aug_clf, aug_k, _cv_method_name, fold, 'fit_time',
                             row['time'], 'N/A', 'N/A')
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv('results.csv', index=False)


def find_best():
    best_dict = dict(dataset=[], fs=[], clf=[], k=[])
    results_path = 'results'
    for dataset in data_loader.available_datasets():
        dataset_results_path = os.path.join(results_path, dataset)
        if not os.path.exists(dataset_results_path):
            continue
        best_index = 0
        best_score = 2
        best_k = 10
        for k in ks:
            k_results_path = os.path.join(dataset_results_path, str(k))
            if not os.path.exists(k_results_path):
                continue
            best_path = os.path.join(k_results_path, 'best.json')
            if not os.path.exists(best_path):
                continue
            with open(best_path, 'r') as f:
                best = json.load(f)
            index = best['index']
            score = best['score']
            if score < best_score:
                best_index = index
                best_score = score
                best_k = k
        k_results_path = os.path.join(dataset_results_path, str(best_k))
        cv_results_path = os.path.join(k_results_path, 'cv_results.csv')
        if not os.path.exists(cv_results_path):
            continue
        cv_results = read_cv_results(cv_results_path)
        best_row = cv_results.iloc[best_index]
        transformer = best_row['param_fs__transformer']
        score_func = best_row['param_fs__transformer__score_func']
        fs_name = fs_method_name(transformer, score_func)
        clf = best_row['param_clf__estimator']
        best_dict['dataset'].append(dataset)
        best_dict['fs'].append(fs_name)
        best_dict['clf'].append(clf)
        best_dict['k'].append(best_k)
    best_df = pd.DataFrame(best_dict)
    best_df.to_csv('best.csv', index=False)


if __name__ == '__main__':
    aggregate_results()
    find_best()
