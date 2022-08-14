import os
import json

import pandas as pd

from functools import partial

from data.experiments import ARFF, bioconductor, Datamicroarray, scikit_feature_datasets
from experiment_utils.metrics import get_metrics


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
    results_dict = {}
    results_path = 'results'
    metric_names = list(get_metrics())
    columns_mapping = {f'mean_test_{metric}': metric for metric in metric_names}
    columns_mapping['mean_fit_time'] = 'time'
    directories = [ARFF, bioconductor, Datamicroarray, scikit_feature_datasets]
    for directory in directories:
        for dataset in directory.load.datasets:
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
            results_dict[dataset] = comparison_df
    return results_dict


def main():
    print('aug')
    for dataset, df in aug().items():
        print(dataset)
        print(df)


if __name__ == '__main__':
    main()
