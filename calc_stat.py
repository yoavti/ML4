import os

import numpy as np

from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

from data import data_loader
from experiment_utils.parameters import ks
from results_processing_utils.read_csv import read_cv_results


ALPHA = 0.05


def gather_scores(metric='ROC_AUC'):
    scores_dict = {}
    results_path = 'results'
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
            cv_results = read_cv_results(cv_results_path, True)
            for _, cv_row in cv_results.iterrows():
                transformer = cv_row['param_fs__transformer']
                clf = cv_row['param_clf__estimator']
                meta_clf = (transformer, clf)
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
        new_intersection = chosen_classifiers.intersection(classifiers_to_add)
        new_fs = {fs for fs, _ in new_intersection}
        if not new_intersection or len(new_fs) < 2:
            continue
        chosen_datasets.add(dataset)
        chosen_classifiers = new_intersection
    scores_dict = {dataset: {classifier: score
                             for classifier, score in classifiers.items()
                             if classifier in chosen_classifiers}
                   for dataset, classifiers in scores_dict.items()
                   if dataset in chosen_datasets}
    return scores_dict, chosen_classifiers


def statistical_tests():
    scores_dict = gather_scores()
    scores_dict, shared_classifiers = keep_shared_classifiers(scores_dict)
    print(shared_classifiers)
    columns_mapping = dict(enumerate(shared_classifiers))
    scores_arr = np.array([[classifier_scores[classifier]
                            for classifier in shared_classifiers]
                           for classifier_scores in scores_dict.values()])
    print(scores_arr)
    statistic, pvalue = friedmanchisquare(*scores_arr.T)
    print(pvalue)
    if pvalue < ALPHA:
        df = posthoc_nemenyi_friedman(scores_arr)
        df = df.rename(columns_mapping)
        df = df.rename(columns_mapping, axis=1)
        df.to_csv('post_hoc.csv')


if __name__ == '__main__':
    statistical_tests()
