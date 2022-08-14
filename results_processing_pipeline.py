from clean import clean
from missing import find_missing
from results_processing import find_best, aggregate_results
from run_aug_experiments import run_aug_experiments
from calc_stat import statistical_tests
from compare import aug, improvement


if __name__ == '__main__':
    clean()
    find_missing()
    find_best()
    run_aug_experiments()
    statistical_tests()
    aug()
    improvement()
    aggregate_results()
