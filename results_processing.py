import os
import json

import pandas as pd

from functools import partial

from data.experiments import ARFF, bioconductor, Datamicroarray, scikit_feature_datasets
from experiment_utils.parameters import ks
from experiment_utils.cv import num_rows, num_folds, cv_method_name
from experiment_utils.metrics import get_metrics


def remove_first_char(s):
    s = s[1:]
    return s


def replace_whitespace(s, sep=','):
    values = s.split()
    s = sep.join(values)
    return s


def remove_quotes(s, sep=','):
    values = s.split(sep)
    values = [value[1:-1] for value in values]
    s = sep.join(values)
    return s


def read_fs(path):
    df = pd.read_csv(path)
    df = df.drop('Unnamed: 0', axis=1)

    df['scores'] = df['scores'].apply(remove_first_char)
    df['scores'] = df['scores'].apply(replace_whitespace)

    df['features'] = df['features'].apply(remove_first_char)
    df['features'] = df['features'].apply(replace_whitespace)
    df['features'] = df['features'].apply(remove_quotes)

    return df


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

    df['param_fs__transformer'] = df['param_fs__transformer'].apply(partial(until, c='('))

    df['param_clf__estimator'] = df['param_clf__estimator'].astype(str)

    df['param_clf__estimator'] = df['param_clf__estimator'].astype(str)
    df['param_clf__estimator'] = df['param_clf__estimator'].apply(partial(until, c='()'))
    return df


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


def main():
    res_dict = {'Dataset Name': [], 'Number of Samples': [], 'Original Number of Features': [],
                'Filtering Algorithm': [], 'Learning Algorithm': [], 'Number of Features Selected': [], 'CV Method': [],
                'Fold': [], 'Measure Type': [], 'Measure Value': [], 'List of Selected Feature Names': [],
                'Selected Feature Scores': []}
    results_path = 'results'
    metric_names = list(get_metrics())
    directories = [ARFF, bioconductor, Datamicroarray, scikit_feature_datasets]
    for directory in directories:
        for dataset, (d, n) in directory.load.dataset_sizes.items():
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
                                         metric_value, fs_scores, fs_features)
                        fs_time = fs_row['times']
                        add_expr_row(res_dict, dataset, n, d, fs_name, clf, k, _cv_method_name, fold, 'fs_time',
                                     fs_time, fs_scores, fs_features)
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
            for _, row in aug_results.iterrows():
                for fold in range(_num_folds):
                    for metric_name in metric_names:
                        metric_value = row[metric_name]
                        add_expr_row(res_dict, dataset, n, d, aug_parameters['fs'], aug_parameters['clf'],
                                     aug_parameters['k'], _cv_method_name, fold, metric_name, metric_value, 'N/A',
                                     'N/A')
                    add_expr_row(res_dict, dataset, n, d, aug_parameters['fs'], aug_parameters['clf'],
                                 aug_parameters['k'], _cv_method_name, fold, 'fit_time', row['time'], 'N/A', 'N/A')
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv('results.csv', index=False)


def find_best():
    best_dict = dict(dataset=[], fs=[], clf=[], k=[])
    results_path = 'results'
    directories = [ARFF, bioconductor, Datamicroarray, scikit_feature_datasets]
    for directory in directories:
        for dataset in directory.load.datasets:
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
                best_path = os.path.join(k_results_path, 'best.csv')
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
    main()

