import os

from data.experiments import ARFF, bioconductor, Datamicroarray, scikit_feature_datasets
from experiment_utils.parameters import ks


def file_empty(path):
    return os.path.getsize(path) == 0


def dir_empty(path):
    return len(os.listdir(path)) == 0


def remove_file_if_empty(path):
    if file_empty(path):
        os.remove(path)


def remove_dir_if_empty(path):
    if dir_empty(path):
        os.rmdir(path)


def clean():
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
                for file in os.listdir(k_results_path):
                    path = os.path.join(k_results_path, file)
                    remove_file_if_empty(path)
                remove_dir_if_empty(k_results_path)
            remove_dir_if_empty(dataset_results_path)


if __name__ == '__main__':
    clean()
