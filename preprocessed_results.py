import os
import shutil

from data import data_loader
from experiment_utils.parameters import named_score_funcs, ks


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def copy_preprocessed():
    fss = list(named_score_funcs) + ['SelectFdr']
    src_path = 'results'
    dst_path = 'preprocessed'
    create_if_not_exists(dst_path)
    for dataset in data_loader.available_datasets():
        dataset_src_path = os.path.join(src_path, dataset)
        if not os.path.exists(dataset_src_path):
            continue
        dataset_dst_path = os.path.join(dst_path, dataset)
        create_if_not_exists(dataset_dst_path)
        for k in ks:
            k_src_path = os.path.join(dataset_src_path, str(k))
            if not os.path.exists(k_src_path):
                continue
            k_dst_path = os.path.join(dataset_dst_path, str(k))
            create_if_not_exists(k_dst_path)
            for fs in fss:
                fs_src_path = os.path.join(k_src_path, f'{fs}.csv')
                if not os.path.exists(fs_src_path):
                    continue
                fs_dst_path = os.path.join(k_dst_path, f'{fs}.csv')
                shutil.copy(fs_src_path, fs_dst_path)


if __name__ == '__main__':
    copy_preprocessed()
